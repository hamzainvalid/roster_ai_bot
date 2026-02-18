import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GEMINI_API_KEY = os.environ["GEM_API"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1
)

app = FastAPI()


# OpenAI-compatible request/response models
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    import time

    # Convert OpenAI format to LangChain messages
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    messages = []
    for msg in request.messages:
        if msg.role == "system":
            messages.append(SystemMessage(content=msg.content))
        elif msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    # Get response from Gemini
    response = llm.invoke(messages)

    # Format as OpenAI-compatible response
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gemini-pro",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response.content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": -1,  # Gemini doesn't provide token counts
            "completion_tokens": -1,
            "total_tokens": -1
        }
    }


@app.get("/v1/models")
async def list_models():
    return {
        "data": [{
            "id": "gemini-2.5-flash",
            "object": "model",
            "created": 1677610602,
            "owned_by": "google"
        }]
    }


# Your existing /ask endpoint
class Question(BaseModel):
    question: str


SYSTEM_PROMPT = """
You generate ONLY SQL.
Table: roster(staff_id, staff_name, date, shift)
Return pure SQL, no markdown, no explanation.
"""


def run_sql(sql):
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        },
        json={"query": sql}
    )
    return r.json()


@app.post("/ask")
def ask(q: Question):
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=q.question)
    ]

    response = llm.invoke(messages)
    sql = response.content

    result = run_sql(sql)

    return {
        "sql": sql,
        "result": result
    }