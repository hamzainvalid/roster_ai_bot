import os
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import OpenAI

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GEMINI_API_KEY = os.environ["GEM_API"]

llm = OpenAI(
    model="gemini-2.5-flash",
    openai_api_key=GEMINI_API_KEY,
    openai_api_base="https://generativelanguage.googleapis.com/v1beta"
)

app = FastAPI()

SYSTEM_PROMPT = """
You are a SQL assistant.

Only answer using database results.

Table: roster(staff_id, staff_name, date, shift)

Shift codes:
D = Day
OFF = Off duty
NP = Night Phone
N = Night
A = Afternoon
AP = Afternoon Phone

For codes you don't know just return codes

Never invent data.
Always query first.
"""

# ================= SUPABASE EXEC =================

def run_sql(sql):
    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        },
        json={"query": sql}
    )
    return response.json()

# ================= OPENAI FORMAT =================

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.0

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):

    # Get latest user message
    user_message = req.messages[-1].content

    # Generate SQL
    prompt = SYSTEM_PROMPT + "\nQuestion: " + user_message
    sql = llm(prompt).strip()

    # Safety check
    if not sql.lower().startswith("select"):
        answer = "Only SELECT queries are allowed."
    else:
        result = run_sql(sql)
        answer = f"SQL:\n{sql}\n\nResult:\n{result}"

    return {
        "id": "chatcmpl-text2sql",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ]
    }
