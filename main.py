import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Added for better compatibility
from pydantic import BaseModel
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
import time
import json

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GEMINI_API_KEY = os.environ["GEM_API"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.1
)

app = FastAPI()

# Add CORS middleware for OpenWebUI compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "model": "gemini-2.5-flash",  # Updated to match your model
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response.content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": -1,
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


# Your existing /ask endpoint with improved formatting
class Question(BaseModel):
    question: str


# Updated SYSTEM_PROMPT to be more explicit
SYSTEM_PROMPT = """
You are a SQL expert. Generate ONLY the SQL query for the given question.
Table schema: roster(staff_id, staff_name, date, shift)

Rules:
- Return pure SQL only
- No markdown formatting
- No explanations
- No backticks
- Date format should be YYYY-MM-DD
- Use single quotes for strings

Example:
Question: Show all staff working on Monday
SQL: SELECT * FROM roster WHERE date = '2024-01-15';
"""

# Second prompt for interpreting results
RESULT_INTERPRETER_PROMPT = """
You are a helpful database assistant. Based on the original question and the SQL query results, provide a natural, conversational answer.

Original Question: {question}
SQL Query Used: {sql}
Query Results: {result}

Provide a clear and concise answer that directly addresses the user's question. If there are no results, explain that politely.
"""


def run_sql(sql):
    """Execute SQL against Supabase"""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
            headers=headers,
            json={"query": sql}
        )

        if r.status_code == 200:
            return r.json()
        else:
            return {"error": f"SQL execution failed: {r.status_code}", "details": r.text}
    except Exception as e:
        return {"error": f"Database connection error: {str(e)}"}


@app.post("/ask")
def ask(q: Question):
    from langchain_core.messages import HumanMessage, SystemMessage

    # Step 1: Generate SQL
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=q.question)
    ]

    response = llm.invoke(messages)
    sql = response.content.strip()

    # Clean up SQL if it has markdown
    sql = sql.replace("```sql", "").replace("```", "").strip()

    # Step 2: Execute SQL
    result = run_sql(sql)

    # Step 3: Generate natural language response
    try:
        interpret_messages = [
            SystemMessage(content="You are a helpful assistant that explains database results in plain English."),
            HumanMessage(content=RESULT_INTERPRETER_PROMPT.format(
                question=q.question,
                sql=sql,
                result=json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
            ))
        ]
        interpretation = llm.invoke(interpret_messages).content
    except Exception as e:
        interpretation = "Could not generate interpretation."

    # Format a beautiful response for OpenWebUI - FIXED SYNTAX
    formatted_response = (
        "### 🔍 Your Question\n"
        f"{q.question}\n\n"
        "### 📝 SQL Query Generated\n"
        "```sql\n"
        f"{sql}\n"
        "```\n\n"
        "### 📊 Query Results\n"
        "```json\n"
        f"{json.dumps(result, indent=2) if isinstance(result, (dict, list)) else result}\n"
        "```\n\n"
        "### 💡 Answer\n"
        f"{interpretation}"
    )

    # Return in the format expected by OpenWebUI's Direct Connection
    return {
        "result": formatted_response
    }


@app.post("/ask-simple")
def ask_simple(q: Question):
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=q.question)
    ]

    response = llm.invoke(messages)
    sql = response.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()

    result = run_sql(sql)

    # Simple formatted response - FIXED SYNTAX
    formatted_response = (
        "**SQL Query:**\n"
        "```sql\n"
        f"{sql}\n"
        "```\n\n"
        "**Result:**\n"
        "```\n"
        f"{json.dumps(result, indent=2) if isinstance(result, (dict, list)) else result}\n"
        "```"
    )

    return {"result": formatted_response}