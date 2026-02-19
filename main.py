import os
import time
import requests
import re
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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
        "HTTP-Referer": "https://your-render-app.onrender.com",
        "X-Title": "Roster Chatbot",
    }
)

MODEL_NAME = "openrouter/free"

# ================= FASTAPI INIT =================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= DATABASE SCHEMA INFO =================

def get_database_schema():
    """Get the actual schema from your roster table"""
    try:
        schema_query = """
        SELECT 
            column_name, 
            data_type 
        FROM information_schema.columns 
        WHERE table_name = 'roster'
        ORDER BY ordinal_position;
        """

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
            headers=headers,
            json={"query": schema_query},
            timeout=10
        )

        if response.status_code == 200:
            columns = response.json()
            if columns is None:
                columns = []

            # Build schema string with explicit note about date being TEXT
            schema_info = "Table: roster("
            col_strings = []
            for col in columns:
                if col and isinstance(col, dict):
                    if col.get('column_name') == 'date':
                        col_strings.append(f"date TEXT (stores dates as YYYY-MM-DD strings)")
                    else:
                        col_strings.append(f"{col.get('column_name', 'unknown')} {col.get('data_type', 'unknown')}")
            schema_info += ", ".join(col_strings)
            schema_info += ")"

            # Get distinct shift codes
            sample_query = "SELECT DISTINCT shift FROM roster WHERE shift IS NOT NULL LIMIT 20;"
            sample_response = requests.post(
                f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
                headers=headers,
                json={"query": sample_query},
                timeout=10
            )

            shift_values = []
            if sample_response.status_code == 200:
                shifts = sample_response.json()
                if shifts and isinstance(shifts, list):
                    shift_values = [s['shift'] for s in shifts if s and isinstance(s, dict) and s.get('shift')]

            return {
                "schema": schema_info,
                "shift_codes": shift_values if shift_values else ["D", "OFF", "NP", "N", "A", "AP"],
                "shift_meanings": {
                    "D": "Day shift",
                    "OFF": "Off duty",
                    "NP": "Night Phone (on-call)",
                    "N": "Night shift",
                    "A": "Afternoon shift",
                    "AP": "Afternoon Phone (on-call)"
                }
            }
    except Exception as e:
        logger.error(f"Error fetching schema: {e}")

    # Fallback schema with correct TEXT date info
    return {
        "schema": "Table: roster(staff_id TEXT, staff_name TEXT, date TEXT (stores dates as YYYY-MM-DD strings), shift TEXT)",
        "shift_codes": ["D", "OFF", "NP", "N", "A", "AP"],
        "shift_meanings": {
            "D": "Day shift",
            "OFF": "Off duty",
            "NP": "Night Phone (on-call)",
            "N": "Night shift",
            "A": "Afternoon shift",
            "AP": "Afternoon Phone (on-call)"
        }
    }


# Get schema at startup
DB_SCHEMA = get_database_schema()
logger.info(f"Database schema loaded: {DB_SCHEMA['schema']}")

# ================= TEXT-TO-SQL PROMPT =================

TEXT_TO_SQL_PROMPT = f"""
You are a SQL expert. Convert natural language questions into PostgreSQL queries.

DATABASE SCHEMA:
{DB_SCHEMA['schema']}

CRITICAL DATA TYPE NOTES (VERY IMPORTANT):
- The 'date' column is TEXT (stores dates as strings in 'YYYY-MM-DD' format)
- The 'shift' column is TEXT
- When comparing with dates, you MUST use string literals with single quotes
- Example: date = '2024-03-25'

SHIFT CODES AND MEANINGS:
{json.dumps(DB_SCHEMA['shift_meanings'], indent=2)}

VALID SHIFT CODES: {', '.join(DB_SCHEMA['shift_codes'])}

DATE HANDLING RULES (CRITICAL):
1. Since date is TEXT, you CANNOT use date arithmetic directly on the column
2. For "today": use CURRENT_DATE::TEXT (converts today's date to text)
3. For "tomorrow": use (CURRENT_DATE + 1)::TEXT
4. For "yesterday": use (CURRENT_DATE - 1)::TEXT
5. For date ranges: compare using BETWEEN with TEXT values (YYYY-MM-DD format works alphabetically)
6. Always use YYYY-MM-DD format in single quotes

EXAMPLES:

Q: "Who is off tomorrow?"
A: SELECT staff_name FROM roster WHERE shift = 'OFF' AND date = (CURRENT_DATE + 1)::TEXT

Q: "What is John's shift today?"
A: SELECT shift FROM roster WHERE staff_name = 'John' AND date = CURRENT_DATE::TEXT

Q: "Who's working night shift this week?"
A: SELECT staff_name, date FROM roster WHERE shift = 'N' AND date BETWEEN CURRENT_DATE::TEXT AND (CURRENT_DATE + 7)::TEXT ORDER BY date

Q: "Show me the roster for Monday"
A: SELECT staff_name, shift FROM roster WHERE date = '2024-03-25' ORDER BY shift

Q: "Is Sarah working on Friday?"
A: SELECT shift FROM roster WHERE staff_name = 'Sarah' AND date = '2024-03-29'

Q: "Who is on afternoon shift today?"
A: SELECT staff_name FROM roster WHERE shift = 'A' AND date = CURRENT_DATE::TEXT

Q: "What's everyone's schedule for next week?"
A: SELECT staff_name, shift, date FROM roster WHERE date BETWEEN CURRENT_DATE::TEXT AND (CURRENT_DATE + 7)::TEXT ORDER BY date, shift

Q: "How many people are on leave today?"
A: SELECT COUNT(*) as count FROM roster WHERE shift = 'OFF' AND date = CURRENT_DATE::TEXT

Now convert this question to SQL. Return ONLY the SQL query, no explanations, no markdown, no semicolons:
"""

# ================= NATURAL LANGUAGE RESPONSE PROMPT =================

NL_RESPONSE_PROMPT = """
You are a friendly HR assistant. Based on the database results, answer the user's question naturally.

User question: {user_question}

SQL query used: {sql_query}

Database results: {results}

SHIFT CODE MEANINGS:
{json.dumps(DB_SCHEMA['shift_meanings'], indent=2)}

Guidelines for response:
1. Be friendly and conversational
2. Use full shift names (e.g., "Day shift" not just "D")
3. For "OFF", say "off duty" or "on leave"
4. Format dates nicely (e.g., "Monday, March 25th")
5. If no results found, politely say so
6. List multiple people in a natural way

Examples of good responses:
- "Tomorrow, John and Sarah are off duty. They'll be back on Tuesday."
- "Mike is working the night shift today."
- "Here's this week's night shift schedule: Monday: David, Tuesday: Emily..."
- "I found 5 people working the afternoon shift today: Alex, Jamie, Taylor, Jordan, and Casey."

Response:
"""


# ================= SUPABASE EXEC =================

def run_sql(sql: str):
    """Execute SQL query on Supabase"""
    try:
        if not sql:
            return {"error": "Empty SQL query"}

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }

        # Remove any trailing semicolons
        clean_sql = sql.rstrip(';')

        logger.info(f"Executing SQL: {clean_sql}")

        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
            headers=headers,
            json={"query": clean_sql},
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"Supabase error: {response.status_code} - {response.text}")
            return []  # Return empty list instead of error dict

        result = response.json()
        # If result is None, return empty list
        if result is None:
            return []
        return result
    except Exception as e:
        logger.error(f"Exception in run_sql: {e}")
        return []  # Return empty list on error


# ================= SQL CLEANING FUNCTION =================

def clean_sql(sql_text):
    """Clean and validate SQL query"""
    if not sql_text:
        return ""

    # Remove markdown code blocks
    if "```sql" in sql_text:
        sql_text = sql_text.split("```sql")[1].split("```")[0]
    elif "```" in sql_text:
        sql_text = sql_text.split("```")[1].split("```")[0]

    # Remove any leading/trailing whitespace
    sql_text = sql_text.strip()

    # Remove trailing semicolons ONLY if they're at the very end
    if sql_text.endswith(';'):
        sql_text = sql_text[:-1]

    # Extract just the SELECT statement if there's extra text
    select_match = re.search(r'(SELECT.*)', sql_text, re.IGNORECASE | re.DOTALL)
    if select_match:
        sql_text = select_match.group(1)

    return sql_text.strip()


# ================= PYDANTIC MODELS =================

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7


# ================= HEALTH CHECK =================

@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database_schema": DB_SCHEMA['schema'],
        "shift_codes": DB_SCHEMA['shift_codes'],
        "deepseek_configured": bool(DEEPSEEK_API_KEY)
    }


# ================= MODEL LISTING ENDPOINTS =================

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """OpenAI compatible model listing"""
    current_time = int(time.time())

    return {
        "object": "list",
        "data": [
            {
                "id": "roster-assistant",
                "object": "model",
                "created": current_time,
                "owned_by": "organization",
                "permission": [{
                    "id": "modelperm-roster",
                    "object": "model_permission",
                    "created": current_time,
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": False,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }],
                "root": "roster-assistant",
                "parent": None
            }
        ]
    }


@app.get("/api/tags")
async def list_ollama_models():
    """Ollama-compatible model listing for OpenWebUI"""
    return {
        "models": [
            {
                "name": "roster-assistant:latest",
                "model": "roster-assistant",
                "modified_at": "2024-01-01T00:00:00Z",
                "size": 1000000000,
                "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "details": {
                    "format": "gguf",
                    "family": "llama",
                    "families": ["llama"],
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            }
        ]
    }


# ================= CHAT ENDPOINT =================

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Get user message
        user_message = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "Hello")
        logger.info(f"User: {user_message}")

        # STEP 1: TEXT-TO-SQL - Generate SQL from natural language
        sql_prompt = TEXT_TO_SQL_PROMPT + f"\nQ: {user_message}\nA:"

        sql_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",
                 "content": "You are a SQL expert. Generate only SQL queries without explanations or semicolons."},
                {"role": "user", "content": sql_prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )

        raw_sql = sql_response.choices[0].message.content.strip()
        logger.info(f"Raw SQL from AI: {raw_sql}")

        # Clean the SQL
        sql = clean_sql(raw_sql)
        logger.info(f"Cleaned SQL: {sql}")

        # STEP 2: EXECUTE SQL or use fallback
        if not sql or not sql.upper().strip().startswith("SELECT"):
            logger.warning(f"Invalid SQL generated, using fallback")

            # Fallback logic for common queries with TEXT date handling
            user_message_lower = user_message.lower()

            if "off" in user_message_lower and "tomorrow" in user_message_lower:
                sql = "SELECT staff_name FROM roster WHERE shift = 'OFF' AND date = (CURRENT_DATE + 1)::TEXT"
            elif "off" in user_message_lower and "today" in user_message_lower:
                sql = "SELECT staff_name FROM roster WHERE shift = 'OFF' AND date = CURRENT_DATE::TEXT"
            elif "off" in user_message_lower and "this week" in user_message_lower:
                sql = "SELECT staff_name, date FROM roster WHERE shift = 'OFF' AND date BETWEEN CURRENT_DATE::TEXT AND (CURRENT_DATE + 7)::TEXT ORDER BY date"
            elif "today" in user_message_lower:
                if "who" in user_message_lower or "everyone" in user_message_lower:
                    sql = "SELECT staff_name, shift FROM roster WHERE date = CURRENT_DATE::TEXT ORDER BY shift"
                else:
                    # Try to extract name
                    words = user_message.split()
                    for word in words:
                        if word and word[0].isupper() and len(word) > 1 and word.lower() not in ['who', 'what', 'when',
                                                                                                 'where', 'why', 'how',
                                                                                                 'today', 'tomorrow',
                                                                                                 'yesterday', 'is',
                                                                                                 'are', 'on', 'for']:
                            sql = f"SELECT shift FROM roster WHERE staff_name = '{word}' AND date = CURRENT_DATE::TEXT"
                            break
            elif "tomorrow" in user_message_lower:
                if "who" in user_message_lower or "everyone" in user_message_lower:
                    sql = "SELECT staff_name, shift FROM roster WHERE date = (CURRENT_DATE + 1)::TEXT ORDER BY shift"
                else:
                    # Try to extract name
                    words = user_message.split()
                    for word in words:
                        if word and word[0].isupper() and len(word) > 1 and word.lower() not in ['who', 'what', 'when',
                                                                                                 'where', 'why', 'how',
                                                                                                 'today', 'tomorrow',
                                                                                                 'yesterday', 'is',
                                                                                                 'are', 'on', 'for']:
                            sql = f"SELECT shift FROM roster WHERE staff_name = '{word}' AND date = (CURRENT_DATE + 1)::TEXT"
                            break
            elif "night" in user_message_lower:
                if "this week" in user_message_lower:
                    sql = "SELECT staff_name, date FROM roster WHERE shift = 'N' AND date BETWEEN CURRENT_DATE::TEXT AND (CURRENT_DATE + 7)::TEXT ORDER BY date"
                else:
                    sql = "SELECT staff_name FROM roster WHERE shift = 'N' AND date = CURRENT_DATE::TEXT"
            elif "afternoon" in user_message_lower:
                if "this week" in user_message_lower:
                    sql = "SELECT staff_name, date FROM roster WHERE shift = 'A' AND date BETWEEN CURRENT_DATE::TEXT AND (CURRENT_DATE + 7)::TEXT ORDER BY date"
                else:
                    sql = "SELECT staff_name FROM roster WHERE shift = 'A' AND date = CURRENT_DATE::TEXT"
            elif "day shift" in user_message_lower or "day" in user_message_lower:
                if "this week" in user_message_lower:
                    sql = "SELECT staff_name, date FROM roster WHERE shift = 'D' AND date BETWEEN CURRENT_DATE::TEXT AND (CURRENT_DATE + 7)::TEXT ORDER BY date"
                else:
                    sql = "SELECT staff_name FROM roster WHERE shift = 'D' AND date = CURRENT_DATE::TEXT"
            else:
                # Generic query
                sql = "SELECT staff_name, shift, date FROM roster WHERE date >= CURRENT_DATE::TEXT ORDER BY date LIMIT 10"

        # Execute the SQL
        results = run_sql(sql)
        logger.info(
            f"Results type: {type(results)}, length: {len(results) if isinstance(results, list) else 'not a list'}")

        # STEP 3: CONVERT RESULTS TO NATURAL LANGUAGE
        # Check if results is a list (successful query)
        if isinstance(results, list):
            if len(results) == 0:
                answer = f"I couldn't find any records matching your query about '{user_message}'. Would you like to ask about something else?"
            else:
                nl_prompt = NL_RESPONSE_PROMPT.format(
                    user_question=user_message,
                    sql_query=sql,
                    results=json.dumps(results, default=str)
                )

                nl_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system",
                         "content": "You are a friendly HR assistant who explains roster data conversationally."},
                        {"role": "user", "content": nl_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )

                answer = nl_response.choices[0].message.content
        else:
            # Results is not a list, something went wrong
            answer = f"I had trouble querying the database. Please try again with a different question."

        # Return OpenAI compatible response
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "roster-assistant",
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
        logger.error(f"Error: {e}", exc_info=True)
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}"
                }
            }]
        }


# if __name__ == "__main__":
#     import uvicorn
#
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run(app, host="0.0.0.0", port=port)