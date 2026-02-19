import os
import time
import requests
import re
import json
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import logging

# ================= LOGGING =================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= ENV VARIABLES =================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# ================= OPENROUTER CLIENT =================

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=DEEPSEEK_API_KEY,
    default_headers={
        "HTTP-Referer": "https://roster-ai-bot.onrender.com",
        "X-Title": "Roster Chatbot",
    }
)

MODEL_NAME = "openrouter/free"

# ================= SHIFT CODES =================

SHIFT_MEANINGS = {
    "D":   "Day shift",
    "N":   "Night shift",
    "A":   "Afternoon shift",
    "NP":  "Night Phone (on-call)",
    "AP":  "Afternoon Phone (on-call)",
    "OFF": "Off duty / Day off"
}

# ================= FASTAPI =================

app = FastAPI(title="Roster Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= SUPABASE SQL RUNNER =================

def run_sql(sql: str) -> list:
    """Execute a SQL query via Supabase RPC and return rows as a list of dicts."""
    if not sql:
        return []
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        }
        clean = sql.strip().rstrip(";")
        logger.info(f"[SQL] {clean}")
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
            headers=headers,
            json={"query": clean},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error(f"Supabase error {resp.status_code}: {resp.text}")
            return []
        result = resp.json()
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.error(f"run_sql exception: {e}")
        return []

# ================= SCHEMA LOADER =================

def load_schema() -> dict:
    try:
        col_resp = run_sql("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'roster'
            ORDER BY ordinal_position
        """)
        if col_resp:
            cols = []
            for c in col_resp:
                name = c.get("column_name", "")
                dtype = c.get("data_type", "")
                if name == "date":
                    cols.append("date TEXT (format: 'YYYY-MM-DD', e.g. '2026-02-18')")
                else:
                    cols.append(f"{name} {dtype}")
            schema_str = "Table: roster(" + ", ".join(cols) + ")"
        else:
            raise ValueError("No columns returned")

        shift_resp = run_sql("SELECT DISTINCT shift FROM roster WHERE shift IS NOT NULL AND shift != '' LIMIT 30")
        shift_codes = [r["shift"] for r in shift_resp if r.get("shift")] if shift_resp else list(SHIFT_MEANINGS.keys())

        return {"schema": schema_str, "shift_codes": shift_codes}
    except Exception as e:
        logger.warning(f"Schema load fallback: {e}")
        return {
            "schema": "Table: roster(staff_id TEXT, staff_name TEXT, date TEXT (format: 'YYYY-MM-DD', e.g. '2026-02-18'), shift TEXT)",
            "shift_codes": list(SHIFT_MEANINGS.keys()),
        }

DB_SCHEMA = load_schema()
logger.info(f"Schema: {DB_SCHEMA['schema']}")

# ================= PROMPT TEMPLATES =================

SYSTEM_SQL = f"""You are a PostgreSQL expert. Convert natural language into a single SQL SELECT query.

SCHEMA:
{DB_SCHEMA['schema']}

SHIFT CODES:
{json.dumps(SHIFT_MEANINGS, indent=2)}

DATE RULES (date column is TEXT, format 'YYYY-MM-DD', e.g. '2026-02-18'):
- Today:      date = CURRENT_DATE::TEXT
- Tomorrow:   date = (CURRENT_DATE + 1)::TEXT
- Yesterday:  date = (CURRENT_DATE - 1)::TEXT
- This week:  date >= CURRENT_DATE::TEXT AND date <= (CURRENT_DATE + 6)::TEXT
- Next week:  date >= (CURRENT_DATE + 7)::TEXT AND date <= (CURRENT_DATE + 13)::TEXT
- This month: date >= date_trunc('month', CURRENT_DATE)::DATE::TEXT AND date < (date_trunc('month', CURRENT_DATE) + INTERVAL '1 month')::DATE::TEXT
- Specific:   date = '2026-02-18'
- Named day (e.g. 'Monday'): use to_char(date::date, 'Day') ILIKE 'Monday%'

RULES:
- Output ONLY the SQL query, nothing else
- No markdown, no backticks, no semicolons, no explanations
- Always use SELECT (never INSERT/UPDATE/DELETE)
- Use ILIKE for case-insensitive name matching
- When asked about a person, use: staff_name ILIKE '%name%'
"""

SYSTEM_NL = f"""You are a friendly, helpful HR assistant for a roster/scheduling system.
Your job is to explain database results in clear, natural, conversational English.

SHIFT MEANINGS:
{json.dumps(SHIFT_MEANINGS, indent=2)}

GUIDELINES:
- Use full shift names (e.g. "Day shift", not "D")
- Format dates nicely (e.g. "February 15th, 2026", not "2026-02-15 00:00:00")
- "OFF" means the person is off duty or has a day off
- Be warm, concise, and professional
- If results are empty, say so helpfully and suggest alternatives
- For lists of people, use natural language (commas + "and")
- For counts: "there is 1 person" vs "there are 3 people"
- If the question seems conversational (not roster-related), just chat naturally
"""

# ================= HELPERS =================

def clean_sql(raw: str) -> str:
    """Strip markdown and extract the SELECT statement."""
    if not raw:
        return ""
    # Strip code fences
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip()
    raw = raw.rstrip(";").strip()
    # Extract from SELECT onward
    m = re.search(r"(SELECT\s.+)", raw, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else raw.strip()


def format_date(date_str: str) -> str:
    """'2026-02-18' or '2026-02-18 00:00:00' -> 'February 18th, 2026'"""
    if not date_str:
        return date_str
    try:
        part = str(date_str).split(" ")[0].split("T")[0]
        d = datetime.strptime(part, "%Y-%m-%d")
        day = d.day
        suffix = "th" if 10 <= day <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return d.strftime(f"%B {day}{suffix}, %Y")
    except Exception:
        return date_str


def is_roster_question(text: str) -> bool:
    """Heuristic: does this message relate to roster/shifts/staff?"""
    keywords = [
        "who", "shift", "working", "roster", "schedule", "off", "today",
        "tomorrow", "week", "night", "day", "afternoon", "on call", "staff",
        "when", "what", "show", "list", "how many", "available", "leave"
    ]
    lower = text.lower()
    return any(k in lower for k in keywords)


def get_available_dates() -> list[str]:
    rows = run_sql("SELECT DISTINCT date FROM roster ORDER BY date LIMIT 10")
    return [format_date(r["date"]) for r in rows if r.get("date")]


# ================= PYDANTIC MODELS =================

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "roster-assistant"
    messages: List[Message]
    temperature: Optional[float] = 0.7

# ================= BUILT-IN COMMANDS =================

HELP_TEXT = """
**🤖 Roster Chatbot — Help**

Ask me anything about the staff roster in plain English. Examples:

**Shifts & Schedules**
- "Who is working today?"
- "Who's on night shift tomorrow?"
- "Show me the roster for next week"
- "What is John's shift on Friday?"

**Off Duty**
- "Who is off tomorrow?"
- "How many people are on leave this week?"

**Specific Staff**
- "Is Sarah working tonight?"
- "What shifts does Mike have this week?"

**Counts**
- "How many people are on day shift today?"

**Debug Commands**
- `debug dates` — Show all dates in the database
- `debug staff` — List all staff members
- `debug shifts` — Show shift code meanings
- `help` — Show this message

Just ask naturally — I'll handle the rest! 😊
""".strip()


async def handle_builtin(cmd: str) -> Optional[str]:
    cmd = cmd.strip().lower()

    if cmd == "help":
        return HELP_TEXT

    if cmd == "debug dates":
        rows = run_sql("SELECT DISTINCT date FROM roster WHERE date NOT LIKE 'Unnamed%' ORDER BY date LIMIT 30")
        if not rows:
            return "No dates found in the database."
        dates = [format_date(r["date"]) for r in rows if r.get("date")]
        count_rows = run_sql("SELECT COUNT(*) as cnt FROM roster WHERE date NOT LIKE 'Unnamed%'")
        total = count_rows[0]["cnt"] if count_rows else "?"
        return f"**📅 Dates in Database** ({total} records)\n\n" + "\n".join(f"• {d}" for d in dates)

    if cmd == "debug staff":
        rows = run_sql("SELECT DISTINCT staff_name FROM roster WHERE staff_name IS NOT NULL AND staff_name != '' ORDER BY staff_name LIMIT 100")
        if not rows:
            return "No staff found in the database."
        names = [r["staff_name"] for r in rows if r.get("staff_name")]
        return f"**👥 Staff Members ({len(names)} total)**\n\n" + ", ".join(names)

    if cmd == "debug shifts":
        lines = [f"• **{k}** — {v}" for k, v in SHIFT_MEANINGS.items()]
        return "**🔄 Shift Code Meanings**\n\n" + "\n".join(lines)

    return None  # Not a built-in command

# ================= CORE PIPELINE =================

def text_to_sql(question: str, history: list[dict]) -> str:
    """Use the LLM to convert a natural language question to SQL."""
    messages = [{"role": "system", "content": SYSTEM_SQL}]
    # Include recent conversation context (last 4 turns) for follow-up questions
    for msg in history[-8:]:
        messages.append(msg)
    messages.append({"role": "user", "content": f"Convert to SQL: {question}"})

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def sql_to_text(question: str, sql: str, results: list, history: list[dict]) -> str:
    """Use the LLM to convert SQL results back to natural language."""
    results_str = json.dumps(results, default=str)

    user_prompt = (
        f"The user asked: \"{question}\"\n\n"
        f"SQL used: {sql}\n\n"
        f"Results ({len(results)} rows): {results_str}\n\n"
        f"Please answer the user's question naturally based on these results."
    )

    messages = [{"role": "system", "content": SYSTEM_NL}]
    for msg in history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_prompt})

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.4,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


def chat_freely(question: str, history: list[dict]) -> str:
    """Handle non-roster conversational messages."""
    messages = [{"role": "system", "content": SYSTEM_NL}]
    for msg in history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()

# ================= CHAT ENDPOINT =================

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        # Extract the latest user message
        user_message = ""
        history = []
        for msg in request.messages:
            if msg.role == "user":
                user_message = msg.content
            history.append({"role": msg.role, "content": msg.content})

        if not user_message:
            user_message = "Hello"

        logger.info(f"User: {user_message}")

        # --- Built-in commands ---
        builtin_response = await handle_builtin(user_message)
        if builtin_response:
            answer = builtin_response
        elif is_roster_question(user_message):
            # --- TEXT → SQL → EXECUTE → NL ---

            # Step 1: Generate SQL
            raw_sql = text_to_sql(user_message, history[:-1])
            sql = clean_sql(raw_sql)
            logger.info(f"Generated SQL: {sql}")

            # Step 2: Validate SQL
            if not sql or not sql.upper().startswith("SELECT"):
                logger.warning("LLM returned invalid SQL, using safe fallback")
                sql = "SELECT staff_name, shift, date FROM roster ORDER BY date LIMIT 20"

            # Step 3: Execute SQL
            results = run_sql(sql)
            logger.info(f"Rows returned: {len(results)}")

            # Step 4: Handle empty results
            if not results:
                available = get_available_dates()
                dates_hint = ""
                if available:
                    dates_hint = f" The roster currently has data for: {', '.join(available[:5])}."
                answer = (
                    f"I couldn't find any matching records for your query.{dates_hint} "
                    f"Try rephrasing, or type `debug dates` to see what's available."
                )
            else:
                # Step 5: SQL results → Natural language
                answer = sql_to_text(user_message, sql, results, history[:-1])
        else:
            # --- Conversational / non-roster message ---
            answer = chat_freely(user_message, history[:-1])

        logger.info(f"Answer: {answer[:120]}...")

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "roster-assistant",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
        }

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
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
                        "content": "Sorry, I ran into an error. Please try again or type `help` for guidance.",
                    },
                    "finish_reason": "stop",
                }
            ],
        }

# ================= UTILITY ENDPOINTS =================

@app.get("/")
@app.get("/health")
async def health():
    count = run_sql("SELECT COUNT(*) as cnt FROM roster")
    records = count[0]["cnt"] if count else "unknown"
    return {
        "status": "healthy",
        "records": records,
        "schema": DB_SCHEMA["schema"],
        "shift_codes": DB_SCHEMA["shift_codes"],
        "model": MODEL_NAME,
    }

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "roster-assistant",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization",
            }
        ],
    }

@app.get("/debug/dates")
async def debug_dates():
    # Total rows in table (no filter)
    total = run_sql("SELECT COUNT(*) as cnt FROM roster")
    total_count = total[0]["cnt"] if total else 0

    # Rows where date looks like a valid date (starts with a digit)
    valid = run_sql("SELECT COUNT(*) as cnt FROM roster WHERE date ~ '^[0-9]'")
    valid_count = valid[0]["cnt"] if valid else 0

    # Rows where date is null or empty
    null_dates = run_sql("SELECT COUNT(*) as cnt FROM roster WHERE date IS NULL OR date = ''")
    null_count = null_dates[0]["cnt"] if null_dates else 0

    # All distinct raw date values (so we can see exactly what's in there)
    raw_distinct = run_sql("SELECT DISTINCT date, COUNT(*) as cnt FROM roster GROUP BY date ORDER BY date LIMIT 50")

    # Valid distinct dates only
    valid_rows = run_sql("SELECT DISTINCT date FROM roster WHERE date ~ '^[0-9]' ORDER BY date LIMIT 50")

    return {
        "total_rows_in_table": total_count,
        "valid_date_rows": valid_count,
        "null_or_empty_date_rows": null_count,
        "invalid_date_rows": total_count - valid_count - null_count,
        "valid_dates_formatted": [format_date(r["date"]) for r in valid_rows if r.get("date")],
        "all_distinct_raw_date_values": raw_distinct,  # raw so you can see exactly what's stored
    }

@app.get("/debug/staff")
async def debug_staff():
    rows = run_sql("SELECT DISTINCT staff_name FROM roster WHERE staff_name IS NOT NULL ORDER BY staff_name")
    return {"staff": [r["staff_name"] for r in rows if r.get("staff_name")]}


@app.get("/debug/inspect")
async def debug_inspect():
    """Full table inspection - reveals actual column names, types, and raw sample rows.
    Use this to diagnose wide-format vs long-format roster data."""

    # All columns with their types
    cols = run_sql("""
        SELECT column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_name = 'roster'
        ORDER BY ordinal_position
    """)

    # First 5 raw rows - no filtering at all
    sample = run_sql("SELECT * FROM roster LIMIT 5")

    # Total row count
    count = run_sql("SELECT COUNT(*) as cnt FROM roster")
    total = count[0]["cnt"] if count else 0

    # Wide vs long heuristic
    num_cols = len(cols) if cols else 0
    suspected_format = "WIDE (each date is its own column - needs unpivoting)" if num_cols > 8 else "LONG (date is a row value - correct format)"

    return {
        "total_rows": total,
        "total_columns": num_cols,
        "suspected_format": suspected_format,
        "columns": cols,
        "sample_rows": sample,
    }

# ================= ENTRYPOINT =================

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run(app, host="0.0.0.0", port=port)