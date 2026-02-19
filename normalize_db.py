import os
import time
import requests
import re
import json
from datetime import datetime, timedelta
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
        "HTTP-Referer": "https://roster-ai-bot.onrender.com",
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

# ================= SHIFT CODE MAPPINGS =================

SHIFT_MEANINGS = {
    "D": "Day shift",
    "OFF": "Off duty",
    "NP": "Night Phone (on-call)",
    "N": "Night shift",
    "A": "Afternoon shift",
    "AP": "Afternoon Phone (on-call)"
}


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

            # Build schema string
            schema_info = "Table: roster("
            col_strings = []
            for col in columns:
                if col and isinstance(col, dict):
                    if col.get('column_name') == 'date':
                        col_strings.append(f"date TEXT (stores dates as 'YYYY-MM-DD HH:MM:SS' strings)")
                    else:
                        col_strings.append(f"{col.get('column_name', 'unknown')} {col.get('data_type', 'unknown')}")
            schema_info += ", ".join(col_strings)
            schema_info += ")"

            # Get distinct shift codes
            sample_query = "SELECT DISTINCT shift FROM roster WHERE shift IS NOT NULL AND shift != '' LIMIT 20;"
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
                "shift_meanings": SHIFT_MEANINGS
            }
    except Exception as e:
        logger.error(f"Error fetching schema: {e}")

    # Fallback schema
    return {
        "schema": "Table: roster(staff_id TEXT, staff_name TEXT, date TEXT (stores dates as 'YYYY-MM-DD HH:MM:SS' strings), shift TEXT)",
        "shift_codes": ["D", "OFF", "NP", "N", "A", "AP"],
        "shift_meanings": SHIFT_MEANINGS
    }


# Get schema at startup
DB_SCHEMA = get_database_schema()
logger.info(f"Database schema loaded: {DB_SCHEMA['schema']}")

# ================= TEXT-TO-SQL PROMPT =================

TEXT_TO_SQL_PROMPT = f"""
You are a SQL expert. Convert natural language questions into PostgreSQL queries.

DATABASE SCHEMA:
{DB_SCHEMA['schema']}

CRITICAL DATA TYPE NOTES:
- The 'date' column is TEXT and stores dates in format: 'YYYY-MM-DD HH:MM:SS' (e.g., '2026-02-01 00:00:00')
- The 'shift' column is TEXT
- When comparing with dates, you need to handle the timestamp format

SHIFT CODES AND MEANINGS:
{json.dumps(DB_SCHEMA['shift_meanings'], indent=2)}

VALID SHIFT CODES: {', '.join(DB_SCHEMA['shift_codes'])}

DATE HANDLING RULES:
1. To compare with today's date: date LIKE (CURRENT_DATE::TEXT || '%')
2. For tomorrow: date LIKE ((CURRENT_DATE + 1)::TEXT || '%')
3. For date ranges: date BETWEEN (start_date || ' 00:00:00') AND (end_date || ' 23:59:59')
4. Always use LIKE with wildcard for partial date matching

EXAMPLES:

Q: "Who is off tomorrow?"
A: SELECT staff_name FROM roster WHERE shift = 'OFF' AND date LIKE ((CURRENT_DATE + 1)::TEXT || '%') ORDER BY staff_name

Q: "What is John's shift today?"
A: SELECT shift FROM roster WHERE staff_name = 'John' AND date LIKE (CURRENT_DATE::TEXT || '%')

Q: "Who's working night shift this week?"
A: SELECT staff_name, date FROM roster WHERE shift = 'N' AND date BETWEEN (CURRENT_DATE::TEXT || ' 00:00:00') AND ((CURRENT_DATE + 7)::TEXT || ' 23:59:59') ORDER BY date

Q: "Show me the roster for Monday"
A: SELECT staff_name, shift FROM roster WHERE date LIKE '2024-03-25%' ORDER BY shift

Q: "Is Sarah working on Friday?"
A: SELECT shift FROM roster WHERE staff_name = 'Sarah' AND date LIKE '2024-03-29%'

Q: "Who is on afternoon shift today?"
A: SELECT staff_name FROM roster WHERE shift = 'A' AND date LIKE (CURRENT_DATE::TEXT || '%') ORDER BY staff_name

Q: "What's everyone's schedule for next week?"
A: SELECT staff_name, shift, date FROM roster WHERE date BETWEEN ((CURRENT_DATE + 7)::TEXT || ' 00:00:00') AND ((CURRENT_DATE + 14)::TEXT || ' 23:59:59') ORDER BY date, shift

Now convert this question to SQL. Return ONLY the SQL query, no explanations, no markdown, no semicolons:
"""

# ================= NATURAL LANGUAGE RESPONSE PROMPT =================

NL_RESPONSE_PROMPT = """
You are a friendly HR assistant. Based on the database results, answer the user's question naturally and conversationally.

User question: {user_question}

SQL query used: {sql_query}

Database results: {results}

SHIFT CODE MEANINGS:
{json.dumps(DB_SCHEMA['shift_meanings'], indent=2)}

RESPONSE GUIDELINES:
1. Be warm, friendly, and professional
2. Use full shift names (e.g., "Day shift" not "D")
3. For "OFF", say "off duty", "on leave", or "taking the day off"
4. Format dates nicely by removing the timestamp (e.g., "February 1st, 2026" not "2026-02-01 00:00:00")
5. For counts, use proper grammar: "is 1 person" vs "are 3 people"
6. When listing multiple people, use commas and "and" naturally
7. If no results found, politely suggest alternatives

Now provide a natural, friendly response:
"""


# ================= SUPABASE EXEC =================

def run_sql(sql: str):
    """Execute SQL query on Supabase"""
    try:
        if not sql:
            return []

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
            return []

        result = response.json()
        return result if result is not None else []
    except Exception as e:
        logger.error(f"Exception in run_sql: {e}")
        return []


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

    sql_text = sql_text.strip()

    # Remove trailing semicolons
    if sql_text.endswith(';'):
        sql_text = sql_text[:-1]

    # Extract SELECT statement
    select_match = re.search(r'(SELECT.*)', sql_text, re.IGNORECASE | re.DOTALL)
    if select_match:
        sql_text = select_match.group(1)

    return sql_text.strip()


# ================= DATE FORMATTING HELPER =================

def format_date(date_str):
    """Convert 'YYYY-MM-DD HH:MM:SS' to friendly format like 'February 1st, 2026'"""
    if not date_str or not isinstance(date_str, str):
        return str(date_str)

    try:
        # Extract just the date part before the space
        if ' ' in date_str:
            date_part = date_str.split(' ')[0]
        else:
            date_part = date_str

        date_obj = datetime.strptime(date_part, '%Y-%m-%d')

        # Add ordinal suffix (1st, 2nd, 3rd, etc.)
        day = date_obj.day
        if 10 <= day <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

        return date_obj.strftime(f'%B {day}{suffix}, %Y')
    except:
        return str(date_str)


# ================= CLEANUP DATABASE ENDPOINT =================

@app.post("/admin/cleanup-database")
async def cleanup_database():
    """Clean up the database by removing unnamed columns and formatting dates"""
    try:
        # Remove rows with "Unnamed" in date column
        cleanup1 = "DELETE FROM roster WHERE date LIKE 'Unnamed%';"
        run_sql(cleanup1)

        # Get cleaned up dates
        dates_sql = "SELECT DISTINCT date FROM roster WHERE date NOT LIKE 'Unnamed%' ORDER BY date LIMIT 20;"
        dates = run_sql(dates_sql)

        formatted_dates = [format_date(d.get('date', '')) for d in dates if d and isinstance(d, dict)]

        return {
            "message": "Database cleaned up",
            "remaining_dates": formatted_dates
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return {"error": str(e)}


# ================= DEBUG ENDPOINTS =================

@app.get("/debug/dates")
async def debug_dates():
    """Check what dates exist in the database"""
    try:
        # Get all unique dates (excluding Unnamed)
        sql = "SELECT DISTINCT date FROM roster WHERE date NOT LIKE 'Unnamed%' ORDER BY date LIMIT 20;"
        results = run_sql(sql)

        # Get today's date
        today_query = "SELECT CURRENT_DATE::TEXT as today;"
        today_result = run_sql(today_query)
        today = today_result[0]['today'] if today_result and len(today_result) > 0 else "unknown"

        # Get tomorrow's date
        tomorrow_query = "SELECT (CURRENT_DATE + 1)::TEXT as tomorrow;"
        tomorrow_result = run_sql(tomorrow_query)
        tomorrow = tomorrow_result[0]['tomorrow'] if tomorrow_result and len(tomorrow_result) > 0 else "unknown"

        # Get total records
        count_query = "SELECT COUNT(*) as count FROM roster;"
        count_result = run_sql(count_query)
        total_records = count_result[0]['count'] if count_result and len(count_result) > 0 else 0

        # Get valid records count (excluding Unnamed)
        valid_count_query = "SELECT COUNT(*) as count FROM roster WHERE date NOT LIKE 'Unnamed%';"
        valid_count_result = run_sql(valid_count_query)
        valid_records = valid_count_result[0]['count'] if valid_count_result and len(valid_count_result) > 0 else 0

        # Format dates for display
        dates_list = []
        if results:
            for d in results:
                if d and isinstance(d, dict) and 'date' in d:
                    dates_list.append(format_date(d['date']))

        return {
            "today": today,
            "tomorrow": tomorrow,
            "total_records": total_records,
            "valid_records": valid_records,
            "invalid_records": total_records - valid_records,
            "dates_in_db": dates_list
        }
    except Exception as e:
        logger.error(f"Debug dates error: {e}")
        return {"error": str(e)}


@app.get("/debug/staff")
async def debug_staff():
    """List all staff members"""
    try:
        sql = "SELECT DISTINCT staff_name FROM roster WHERE staff_name IS NOT NULL AND staff_name != '' ORDER BY staff_name LIMIT 50;"
        results = run_sql(sql)
        staff_list = [s['staff_name'] for s in results if
                      s and isinstance(s, dict) and s.get('staff_name')] if results else []
        return {"staff_members": staff_list}
    except Exception as e:
        logger.error(f"Debug staff error: {e}")
        return {"error": str(e)}


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
    # Get quick stats
    count_query = "SELECT COUNT(*) as count FROM roster WHERE date NOT LIKE 'Unnamed%';"
    count_result = run_sql(count_query)
    valid_records = count_result[0]['count'] if count_result and len(count_result) > 0 else 0

    return {
        "status": "healthy",
        "database": {
            "schema": DB_SCHEMA['schema'],
            "valid_records": valid_records
        },
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
                "owned_by": "organization"
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

        # Handle debug commands
        if user_message.lower() == "debug dates":
            debug_info = await debug_dates()
            if "error" in debug_info:
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": f"Error getting dates: {debug_info['error']}"
                        }
                    }]
                }

            dates_str = "\n".join([f"• {d}" for d in debug_info.get('dates_in_db', [])])
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"📊 **Database Debug Info**\n\n"
                                   f"Total Records: {debug_info.get('total_records')}\n"
                                   f"Valid Records: {debug_info.get('valid_records')}\n"
                                   f"Invalid Records: {debug_info.get('invalid_records')}\n\n"
                                   f"**Available Dates:**\n{dates_str}"
                    }
                }]
            }

        if user_message.lower() == "debug staff":
            debug_info = await debug_staff()
            if "error" in debug_info:
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": f"Error getting staff: {debug_info['error']}"
                        }
                    }]
                }

            staff_list = ", ".join(debug_info.get('staff_members', []))
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"👥 **Staff Members**\n\n{staff_list}"
                    }
                }]
            }

        if user_message.lower() == "help":
            help_text = """
**🤖 Roster Chatbot - Help**

You can ask me anything about the staff roster! Here are some examples:

**Individual Staff:**
- "What is John's shift today?"
- "Is Sarah working tomorrow?"
- "What are Mike's shifts this week?"

**Shift-Based:**
- "Who is working night shift tonight?"
- "Who's on day shift tomorrow?"
- "How many people are on afternoon shift today?"

**Off Duty:**
- "Who is off tomorrow?"
- "Who's on leave this week?"

**Date Range:**
- "What's the roster for today?"
- "Show me next week's schedule"

**Debug Commands:**
- "debug dates" - Show available dates
- "debug staff" - List all staff members
- "help" - Show this help message

Just ask naturally and I'll help you out! 😊
"""
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": help_text
                    }
                }]
            }

        if user_message.lower() == "cleanup database":
            result = await cleanup_database()
            if "error" in result:
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": f"Error cleaning database: {result['error']}"
                        }
                    }]
                }

            dates_str = ", ".join(result.get('remaining_dates', []))
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"🧹 Database cleanup completed! Remaining dates: {dates_str}"
                    }
                }]
            }

        # Check if there's valid data
        check_data_sql = "SELECT COUNT(*) as count FROM roster WHERE date NOT LIKE 'Unnamed%';"
        count_result = run_sql(check_data_sql)
        valid_records = count_result[0]['count'] if count_result and len(count_result) > 0 else 0

        if valid_records == 0:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "📭 The roster database contains no valid records. Try running 'cleanup database' first or check your data upload."
                    }
                }]
            }

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

        # STEP 2: EXECUTE SQL
        if not sql or not sql.upper().strip().startswith("SELECT"):
            logger.warning(f"Invalid SQL generated, using fallback")

            # Simple fallback for common queries
            user_message_lower = user_message.lower()

            if "off" in user_message_lower and "tomorrow" in user_message_lower:
                sql = "SELECT staff_name FROM roster WHERE shift = 'OFF' AND date LIKE ((CURRENT_DATE + 1)::TEXT || '%') ORDER BY staff_name"
            elif "off" in user_message_lower and "today" in user_message_lower:
                sql = "SELECT staff_name FROM roster WHERE shift = 'OFF' AND date LIKE (CURRENT_DATE::TEXT || '%') ORDER BY staff_name"
            elif "today" in user_message_lower and "who" in user_message_lower:
                sql = "SELECT staff_name, shift FROM roster WHERE date LIKE (CURRENT_DATE::TEXT || '%') ORDER BY shift"
            elif "tomorrow" in user_message_lower and "who" in user_message_lower:
                sql = "SELECT staff_name, shift FROM roster WHERE date LIKE ((CURRENT_DATE + 1)::TEXT || '%') ORDER BY shift"
            elif "night" in user_message_lower and "today" in user_message_lower:
                sql = "SELECT staff_name FROM roster WHERE shift = 'N' AND date LIKE (CURRENT_DATE::TEXT || '%') ORDER BY staff_name"
            elif "afternoon" in user_message_lower and "today" in user_message_lower:
                sql = "SELECT staff_name FROM roster WHERE shift = 'A' AND date LIKE (CURRENT_DATE::TEXT || '%') ORDER BY staff_name"
            elif "day" in user_message_lower and "today" in user_message_lower:
                sql = "SELECT staff_name FROM roster WHERE shift = 'D' AND date LIKE (CURRENT_DATE::TEXT || '%') ORDER BY staff_name"
            else:
                # Show recent dates
                sql = "SELECT staff_name, shift, date FROM roster WHERE date NOT LIKE 'Unnamed%' ORDER BY date LIMIT 10"

        logger.info(f"Final SQL: {sql}")

        # Execute the SQL
        results = run_sql(sql)
        logger.info(f"Results count: {len(results) if isinstance(results, list) else 0}")

        # STEP 3: CONVERT RESULTS TO NATURAL LANGUAGE
        if isinstance(results, list):
            if len(results) == 0:
                # Get available dates to suggest
                date_sql = "SELECT DISTINCT date FROM roster WHERE date NOT LIKE 'Unnamed%' ORDER BY date LIMIT 5;"
                date_results = run_sql(date_sql)

                if date_results and len(date_results) > 0:
                    available_dates = []
                    for d in date_results:
                        if d and isinstance(d, dict) and 'date' in d:
                            available_dates.append(format_date(d['date']))

                    if available_dates:
                        dates_str = ", ".join(available_dates)
                        answer = f"I couldn't find any records matching your query. The available dates in the system are: {dates_str}. Would you like to check one of these dates instead?"
                    else:
                        answer = f"I couldn't find any records matching your query about '{user_message}'. Try asking 'debug dates' to see what dates are available."
                else:
                    answer = f"I couldn't find any records matching your query about '{user_message}'. Try asking 'help' to see example queries or 'debug dates' to see what dates are available."
            else:
                # Format results for the prompt - ensure it's JSON serializable
                try:
                    results_json = json.dumps(results, default=str)
                except:
                    results_json = str(results)

                nl_prompt = NL_RESPONSE_PROMPT.format(
                    user_question=user_message,
                    sql_query=sql,
                    results=results_json
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
            answer = "I had trouble querying the database. Please try again with a different question."

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
                    "content": f"I encountered an error. Please try again or ask for 'help'."
                }
            }]
        }


# if __name__ == "__main__":
#     import uvicorn
#
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run(app, host="0.0.0.0", port=port)