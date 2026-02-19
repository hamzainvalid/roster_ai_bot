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

        # Step 1: Generate SQL from user question
        sql_prompt = SYSTEM_PROMPT + "\n\nGenerate ONLY a SQL query for this question: " + user_message

        response = model.generate_content(sql_prompt)

        if not response.text:
            return {"error": "Gemini returned empty response."}

        sql = response.text.strip()

        # Basic sanitization
        sql_lower = sql.lower()

        if not sql_lower.startswith("select"):
            answer = "I can only look up information, not make changes to the database."
        elif any(word in sql_lower for word in ["insert", "update", "delete", "drop", "alter"]):
            answer = "I'm sorry, I can only read information from the database, not modify it."
        else:
            # Execute SQL
            result = run_sql(sql)

            # Step 2: Convert results to natural language
            if result and isinstance(result, list) and len(result) > 0:
                # Create a prompt to convert the results to natural language
                conversion_prompt = f"""
                Based on this user question: "{user_message}"

                And these database results: {result}

                Provide a friendly, natural language response that answers the user's question.
                Use the shift code meanings: D=Day, OFF=Off duty, N=Night, A=Afternoon, NP=Night Phone, AP=Afternoon Phone.
                If the results show multiple people, list them nicely.
                Keep it conversational but informative.
                """

                natural_response = model.generate_content(conversion_prompt)
                answer = natural_response.text if natural_response.text else str(result)
            else:
                answer = f"I couldn't find any information matching your question about '{user_message}'. Would you like to ask about something else?"

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
