import os
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import OpenAI
import logging

# ================= LOGGING CONFIG =================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= ENV VARIABLES =================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# ================= DEEPSEEK CONFIG =================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=DEEPSEEK_API_KEY,
    default_headers={
        "HTTP-Referer": "https://roster-ai-bot.onrender.com",  # Replace with your actual Render URL
        "X-Title": "Roster Chatbot",
    }
)

# Use the free auto-router - this will use whatever free model is available
MODEL_NAME = "openrouter/free"

# ================= FASTAPI INIT =================

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OpenWebUI might be on different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= SYSTEM PROMPT =================

SYSTEM_PROMPT = """
You are a friendly and helpful HR assistant specializing in staff rosters. Your role is to answer questions about staff schedules in a natural, conversational way.

DATABASE SCHEMA:
Table: roster(staff_id, staff_name, date, shift)

SHIFT CODES:
- D = Day shift
- OFF = Off duty / Day off
- NP = Night Phone (on-call)
- N = Night shift
- A = Afternoon shift
- AP = Afternoon Phone (on-call)

YOUR TASK:
1. First, understand what the user is asking about the roster
2. Generate a SQL query to fetch the exact data needed
3. After getting the SQL results, convert them into a natural, friendly response

RULES:
- Always convert the raw data into conversational English
- Use staff names not IDs in your responses
- For date queries, understand relative terms like "today", "tomorrow", "this week"
- When someone is off duty, say they're "off" or "on leave" not "OFF"
- Expand shift codes to their full meanings
- If no data is found, politely say so and suggest alternatives
- Keep responses concise but friendly
- Never make up or invent data

Remember: You're not just generating SQL - you're having a conversation about staff schedules. Be helpful, friendly, and always respond in plain English with the actual data from the database.
"""


# ================= SUPABASE EXEC =================

def run_sql(sql: str):
    """Execute SQL query on Supabase"""
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
            headers=headers,
            json={"query": sql},
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"Supabase error: {response.status_code} - {response.text}")
            return {"error": f"Database error: {response.text}"}

        return response.json()
    except Exception as e:
        logger.error(f"Exception in run_sql: {e}")
        return {"error": str(e)}


# ================= PYDANTIC MODELS (OpenAI Compatible) =================

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


# ================= HEALTH CHECK =================

@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "deepseek_configured": bool(DEEPSEEK_API_KEY and client),
    }


# ================= OPENAI COMPATIBLE ENDPOINTS =================

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """Return available models (OpenAI compatible format)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "roster-assistant",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization",
                "permission": [],
                "root": "roster-assistant",
                "parent": None
            },
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepseek",
                "permission": [],
                "root": MODEL_NAME,
                "parent": None
            }
        ]
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Main chat endpoint (OpenAI compatible format)"""
    try:
        logger.info(f"Received request with {len(request.messages)} messages")

        if not client:
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I'm sorry, but the AI service is not properly configured. Please check the server logs."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

        # Get the last user message
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break

        if not user_message:
            user_message = "Hello"

        logger.info(f"User message: {user_message}")

        # Step 1: Generate SQL
        sql_prompt = f"""{SYSTEM_PROMPT}

Based on the user's question below, generate ONLY a SQL query (SELECT statement) that would fetch the required data from the roster table. 
Return ONLY the SQL query, no explanations or additional text.

User question: {user_message}

SQL query:"""

        sql_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "You are a SQL expert. Generate precise SQL queries based on user questions."},
                {"role": "user", "content": sql_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )

        sql = sql_response.choices[0].message.content.strip()

        # Clean SQL
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0].strip()

        logger.info(f"Generated SQL: {sql}")

        # Validate and execute SQL
        sql_lower = sql.lower()
        if not sql_lower.startswith("select"):
            answer = "I can only look up information, not make changes to the database."
        elif any(word in sql_lower for word in ["insert", "update", "delete", "drop", "alter"]):
            answer = "I'm sorry, I can only read information from the database, not modify it."
        else:
            result = run_sql(sql)
            logger.info(f"SQL result: {str(result)[:200]}...")

            if result and isinstance(result, list) and len(result) > 0:
                conversion_prompt = f"""
                User question: "{user_message}"

                Database results: {result}

                Provide a friendly, natural language response that answers the user's question.
                Use the shift code meanings: D=Day shift, OFF=Off duty, N=Night shift, A=Afternoon shift, NP=Night Phone, AP=Afternoon Phone.
                If the results show multiple people, list them in a friendly way.
                Be conversational but informative.

                Response:"""

                natural_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system",
                         "content": "You are a friendly HR assistant who explains roster data conversationally."},
                        {"role": "user", "content": conversion_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                answer = natural_response.choices[0].message.content
            else:
                answer = f"I couldn't find any information matching your question about '{user_message}'. Would you like to ask about something else?"

        # Return OpenAI compatible response
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,  # Approximate
                "completion_tokens": 50,  # Approximate
                "total_tokens": 150
            }
        }

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model if request else "unknown",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"I encountered an error: {str(e)}"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


# if __name__ == "__main__":
#     import uvicorn
#
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run(app, host="0.0.0.0", port=port)