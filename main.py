import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import time


# ================= ENV VARIABLES =================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API"]  # Your OpenRouter or DeepSeek API key

# ================= DEEPSEEK CONFIG =================

# Using OpenRouter's free DeepSeek endpoint (recommended)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=DEEPSEEK_API_KEY,
)

# Alternative: Direct DeepSeek API (requires top-up but has free quota)
# client = OpenAI(
#     base_url="https://api.deepseek.com/v1",
#     api_key=DEEPSEEK_API_KEY,
# )

# Model to use (free on OpenRouter)
MODEL_NAME = "deepseek/deepseek-r1:free"  # Free tier on OpenRouter
# Alternative: "deepseek/deepseek-chat" for paid version

# ================= FASTAPI INIT =================

app = FastAPI()

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

EXAMPLES OF GOOD RESPONSES:
User: "Who is off tomorrow?"
SQL: SELECT staff_name FROM roster WHERE shift = 'OFF' AND date = CURRENT_DATE + 1
Response: "Tomorrow, the following staff members are off duty: [names]. They'll be enjoying their day off!"

User: "What's John's shift today?"
SQL: SELECT shift FROM roster WHERE staff_name = 'John' AND date = CURRENT_DATE
Response: "John is working the [Day/Afternoon/Night] shift today." or "John is off duty today."

User: "Who's working night shift this week?"
SQL: SELECT staff_name FROM roster WHERE shift = 'N' AND date BETWEEN CURRENT_DATE AND CURRENT_DATE + 7
Response: "This week's night shift staff are: [names]. They'll be working through the night."

User: "Show me the roster for Monday"
SQL: SELECT staff_name, shift FROM roster WHERE date = '2024-01-15' ORDER BY shift
Response: "Here's the roster for Monday, January 15th:
- Day shift: [names]
- Afternoon shift: [names]
- Night shift: [names]
- Off duty: [names]"

Remember: You're not just generating SQL - you're having a conversation about staff schedules. Be helpful, friendly, and always respond in plain English with the actual data from the database.
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
                "id": "roster-assistant",
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

        # Step 1: Generate SQL from user question
        sql_prompt = f"""{SYSTEM_PROMPT}

Based on the user's question below, generate ONLY a SQL query (SELECT statement) that would fetch the required data from the roster table. 
Return ONLY the SQL query, no explanations or additional text.

User question: {user_message}

SQL query:"""

        # Call DeepSeek for SQL generation
        sql_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "You are a SQL expert. Generate precise SQL queries based on user questions."},
                {"role": "user", "content": sql_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent SQL generation
            max_tokens=500
        )

        if not sql_response.choices or not sql_response.choices[0].message.content:
            return {"error": "DeepSeek returned empty response."}

        sql = sql_response.choices[0].message.content.strip()

        # Clean up SQL if it contains markdown or extra text
        if "```sql" in sql:
            sql = sql.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0].strip()

        # Basic sanitization
        sql_lower = sql.lower()

        if not sql_lower.startswith("select"):
            answer = "I can only look up information, not make changes to the database."
        elif any(word in sql_lower for word in ["insert", "update", "delete", "drop", "alter", "create", "truncate"]):
            answer = "I'm sorry, I can only read information from the database, not modify it."
        else:
            # Execute SQL
            result = run_sql(sql)

            # Step 2: Convert results to natural language
            if result and isinstance(result, list) and len(result) > 0:
                # Create a prompt to convert the results to natural language
                conversion_prompt = f"""
                User question: "{user_message}"

                Database results: {result}

                SQL query used: {sql}

                Provide a friendly, natural language response that answers the user's question based on these database results.

                Guidelines:
                - Use the shift code meanings: D=Day shift, OFF=Off duty, N=Night shift, A=Afternoon shift, NP=Night Phone, AP=Afternoon Phone
                - If results show multiple people, list them in a friendly way
                - Format dates in a readable format (e.g., "Monday, January 15th")
                - Be conversational but informative
                - If the results show shift times, mention them naturally
                - Keep it concise but complete

                Response:"""

                natural_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system",
                         "content": "You are a friendly HR assistant who explains roster data conversationally."},
                        {"role": "user", "content": conversion_prompt}
                    ],
                    temperature=0.3,  # Slightly higher for natural conversation
                    max_tokens=500
                )

                answer = natural_response.choices[0].message.content if natural_response.choices else str(result)
            else:
                answer = f"I couldn't find any information matching your question about '{user_message}'. Would you like to ask about something else? For example, you could ask 'Who's working today?' or 'What's Sarah's shift tomorrow?'"

        # OpenAI compatible response
        return {
            "id": "chatcmpl-roster-bot",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
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
                "prompt_tokens": sql_response.usage.prompt_tokens if hasattr(sql_response, 'usage') else 0,
                "completion_tokens": sql_response.usage.completion_tokens if hasattr(sql_response, 'usage') else 0,
                "total_tokens": sql_response.usage.total_tokens if hasattr(sql_response, 'usage') else 0
            }
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "I encountered an error processing your request. Please try again."
        }


# ================= RENDER PORT SUPPORT =================
#
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=port)
