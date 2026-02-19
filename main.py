import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai

# ================= ENV VARIABLES =================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GEMINI_API_KEY = os.environ["GEM_API"]

# ================= GEMINI CONFIG =================

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ================= FASTAPI INIT =================

app = FastAPI()

# ================= SYSTEM PROMPT =================

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

def run_sql(sql: str):
    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        },
        json={"query": sql}
    )

    if response.status_code != 200:
        return {"error": response.text}

    return response.json()

# ================= OPENAI COMPATIBLE FORMAT =================

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.0

# ================= MODEL LIST =================

@app.get("/v1/models")
@app.get("/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "text2sql-roster",
                "object": "model",
                "owned_by": "organization"
            }
        ]
    }

# ================= CHAT ENDPOINT =================

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat(req: ChatRequest):
    try:
        user_message = req.messages[-1].content

        # Build Gemini prompt
        prompt = SYSTEM_PROMPT + "\n\nQuestion: " + user_message

        # Call Gemini
        response = model.generate_content(prompt)

        if not response.text:
            return {"error": "Gemini returned empty response."}

        sql = response.text.strip()

        # Basic sanitization
        sql_lower = sql.lower()

        if not sql_lower.startswith("select"):
            answer = "Only SELECT queries are allowed."
        elif any(word in sql_lower for word in ["insert", "update", "delete", "drop", "alter"]):
            answer = "Dangerous query detected. Only SELECT allowed."
        else:
            result = run_sql(sql)
            answer = str(result)

        # OpenAI compatible response
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

    except Exception as e:
        return {
            "error": str(e)
        }

# ================= RENDER PORT SUPPORT =================
#
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)
