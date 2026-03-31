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
import calendar

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

MODEL_NAME = "deepseek/deepseek-chat:free"

# ================= SHIFT CODES =================

SHIFT_MEANINGS = {
    "A": "Afternoon shift (14:00-23:00)",
    "A*": "Afternoon shift (modified/extended)",
    "AP": "Afternoon Phone - on-call afternoon shift",
    "AP/OT": "Afternoon Phone / Overtime",
    "BL": "Bonus Leave",
    "D": "Day shift (06:00-15:00)",
    "D*": "Day shift (modified/extended)",
    "D-OJT": "Day shift - On-the-Job Training",
    "D.OT": "Day shift with Overtime",
    "DIL": "Day in Lieu",
    "DP": "Day Phone - on-call day shift",
    "DT": "Duty Travel",
    "FL": "Family Leave",
    "N": "Night shift (16:00-22:00)",
    "NP": "Night Phone - on-call night shift",
    "OFF": "Day off / Rest day",
    "OFF-OT": "Day off with Overtime",
    "OT": "Overtime",
    "OT-AP": "Overtime - Afternoon Phone",
    "OT-DP": "Overtime - Day Phone",
    "PH": "Public Holiday",
    "PV": "Pre-approved Vacation",
    "Sick": "Sick (unplanned)",
    "SL": "Sick Leave",
    "T": "Training",
    "TRNG": "Training",
    "V": "Annual Leave (Vacation)",
    "V-Sick": "Vacation converted to Sick Leave",
    "Vsick": "Vacation converted to Sick Leave",
    "sick": "Sick (unplanned)",
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


# ================= SUPABASE HELPERS =================

def _headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def run_sql(sql: str) -> list:
    """Run a SELECT query via run_sql RPC."""
    if not sql:
        return []
    try:
        clean = sql.strip().rstrip(";")
        logger.info(f"[SQL] {clean[:200]}")
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
            headers=_headers(),
            json={"query": clean},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error(f"Supabase error {resp.status_code}: {resp.text[:300]}")
            return []
        result = resp.json()
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.error(f"run_sql error: {e}")
        return []


def rest_update(table: str, staff_name: str, date: str, new_shift: str) -> bool:
    """Update a single shift via REST PATCH."""
    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={**_headers(), "Prefer": "return=representation"},
            params={"staff_name": f"eq.{staff_name}", "date": f"eq.{date}"},
            json={"shift": new_shift},
            timeout=30,
        )
        logger.info(f"[UPDATE] {table} | {staff_name} | {date} → {new_shift} | status={resp.status_code}")
        return resp.status_code in (200, 204)
    except Exception as e:
        logger.error(f"rest_update error: {e}")
        return False


def rest_get_month(table: str) -> list:
    """Fetch all rows for a monthly table via REST."""
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_headers(),
            params={"select": "staff_name,staff_id,date,shift", "order": "date,staff_name"},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        logger.error(f"rest_get_month error: {e}")
        return []


def table_exists(table: str) -> bool:
    """Check if a monthly table exists."""
    rows = run_sql(f"""
        SELECT 1 FROM information_schema.tables
        WHERE table_name = '{table}' AND table_schema = 'public'
    """)
    return bool(rows)


def month_table(year: int, month: int) -> str:
    return f"roster_{year:04d}_{month:02d}"


def get_available_months() -> list[str]:
    rows = run_sql("""
        SELECT table_name FROM information_schema.tables
        WHERE table_name LIKE 'roster_%_%' AND table_schema = 'public'
        ORDER BY table_name
    """)
    return [r["table_name"] for r in rows if r.get("table_name")]


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
                    cols.append("date TEXT (format: 'YYYY-MM-DD')")
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
            "schema": "Table: roster(id BIGINT, staff_name TEXT, staff_id TEXT, date TEXT (format: 'YYYY-MM-DD'), shift TEXT)",
            "shift_codes": list(SHIFT_MEANINGS.keys()),
        }


DB_SCHEMA = load_schema()
logger.info(f"Schema: {DB_SCHEMA['schema']}")

# ================= PROMPT TEMPLATES =================

SYSTEM_SQL = f"""You are a PostgreSQL expert. Convert natural language into a single SQL SELECT query.

SCHEMA:
Main table: roster(id, staff_name TEXT, staff_id TEXT, date TEXT 'YYYY-MM-DD', shift TEXT)
Monthly tables: roster_YYYY_MM  e.g. roster_2026_03 — same columns, only that month's data.

SHIFT CODES:
{json.dumps(SHIFT_MEANINGS, indent=2)}

DATE RULES (date column is TEXT format 'YYYY-MM-DD'):
- Today:      date = CURRENT_DATE::TEXT
- Tomorrow:   date = (CURRENT_DATE + 1)::TEXT
- This week:  date >= CURRENT_DATE::TEXT AND date <= (CURRENT_DATE + 6)::TEXT
- Next week:  date >= (CURRENT_DATE + 7)::TEXT AND date <= (CURRENT_DATE + 13)::TEXT
- A specific month: use the monthly table roster_YYYY_MM instead of roster
- Named day:  to_char(date::date, 'Day') ILIKE 'Monday%'

TABLE SELECTION RULES:
- For queries about a specific month (e.g. "March 2026", "next month"): use roster_YYYY_MM
- For queries spanning multiple months or no specific month: use roster
- Current date context: {datetime.now().strftime('%Y-%m-%d')} (use this to resolve "next month", "this month" etc.)

RULES:
- Output ONLY the SQL query, nothing else
- No markdown, no backticks, no semicolons, no explanations
- Always SELECT (never INSERT/UPDATE/DELETE)
- Use ILIKE for case-insensitive name matching
"""

SYSTEM_NL = f"""You are a friendly, helpful HR assistant for a roster/scheduling system.
Your job is to explain database results and changes in clear, natural, conversational English.

SHIFT MEANINGS:
{json.dumps(SHIFT_MEANINGS, indent=2)}

GUIDELINES:
- Use full shift names (e.g. "Day shift", not "D")
- Format dates nicely (e.g. "March 15th, 2026")
- "OFF" means the person is off duty
- Be warm, concise, and professional
- For lists, use natural language with commas and "and"
- For counts: "there is 1 person" vs "there are 3 people"
- When reporting a change, clearly state: who, what date, old shift → new shift
- When showing a month roster, format it as a clean readable summary
"""

SYSTEM_UPDATE = f"""You are a roster update parser. Extract shift change details from natural language.

Today is {datetime.now().strftime('%Y-%m-%d')}.

Your job: parse the user's message and return a JSON object with these fields:
{{
  "action": "update",
  "staff_name": "<full or partial name>",
  "date": "<YYYY-MM-DD>",
  "new_shift": "<shift code>",
  "show_month": true/false
}}

SHIFT CODES: {json.dumps(list(SHIFT_MEANINGS.keys()))}

RULES:
- "next Monday" / "this Friday" etc → resolve to actual YYYY-MM-DD dates
- If user says "change to day shift" → new_shift = "D"
- If user says "give him a day off" → new_shift = "OFF"
- If user says "put on afternoon" → new_shift = "A"
- set show_month = true if user wants to see the updated month roster
- If you cannot confidently parse all fields, return {{"action": "unclear"}}
- Return ONLY the JSON object, no explanation
"""


# ================= HELPERS =================

def clean_sql(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip()
    raw = raw.rstrip(";").strip()
    m = re.search(r"(SELECT\s.+)", raw, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else raw.strip()


def format_date(date_str: str) -> str:
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


def is_update_intent(text: str) -> bool:
    keywords = [
        "change", "update", "modify", "swap", "switch", "move",
        "set", "put", "assign", "replace", "edit", "reschedule",
        "give him", "give her", "make him", "make her"
    ]
    lower = text.lower()
    return any(k in lower for k in keywords)


def is_roster_question(text: str) -> bool:
    keywords = [
        "who", "shift", "working", "roster", "schedule", "off", "today",
        "tomorrow", "week", "night", "day", "afternoon", "on call", "staff",
        "when", "what", "show", "list", "how many", "available", "leave",
        "month", "next month", "this month"
    ]
    lower = text.lower()
    return any(k in lower for k in keywords)


def get_available_dates() -> list[str]:
    rows = run_sql("SELECT DISTINCT date FROM roster ORDER BY date LIMIT 10")
    return [format_date(r["date"]) for r in rows if r.get("date")]


def format_month_roster(rows: list, year: int, month: int) -> str:
    """Format monthly roster rows into a readable markdown table."""
    if not rows:
        return "No data found."
    month_name = datetime(year, month, 1).strftime("%B %Y")
    days = sorted(set(r["date"] for r in rows))
    staff = sorted(set(r["staff_name"] for r in rows))
    lookup = {(r["staff_name"], r["date"]): r["shift"] for r in rows}

    lines = [f"**{month_name} Roster**\n"]
    for s in staff:
        shifts = []
        for d in days:
            shift = lookup.get((s, d), "-")
            day_num = d.split("-")[2].lstrip("0")
            shifts.append(f"{day_num}:{shift}")
        lines.append(f"**{s}**: " + "  ".join(shifts))
    return "\n".join(lines)


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
**Roster Chatbot — Help**

**Read the roster:**
- "Who is working today?"
- "Who's on night shift tomorrow?"
- "Show me the roster for March 2026"
- "What is Sara's shift on February 18th?"
- "Who is off next week?"

**Make changes:**
- "Change Sara's shift on March 5th to Day shift"
- "Give Muawia a day off on April 10th"
- "Update Gatot's shift on March 20 to Afternoon"
- "Switch Qadir's shift on the 15th to OFF and show me the March roster"

**Debug commands:**
- `debug dates` — show available dates
- `debug staff` — list all staff
- `debug shifts` — show shift code meanings
- `debug months` — show all monthly tables
- `help` — this message
""".strip()


async def handle_builtin(cmd: str) -> Optional[str]:
    cmd_lower = cmd.strip().lower()
    if cmd_lower == "help":
        return HELP_TEXT
    if cmd_lower == "debug dates":
        rows = run_sql("SELECT DISTINCT date FROM roster ORDER BY date LIMIT 30")
        dates = [format_date(r["date"]) for r in rows if r.get("date")]
        return "**Dates in DB:**\n" + "\n".join(f"• {d}" for d in dates)
    if cmd_lower == "debug staff":
        rows = run_sql("SELECT DISTINCT staff_name FROM roster ORDER BY staff_name")
        names = [r["staff_name"] for r in rows if r.get("staff_name")]
        return f"**Staff ({len(names)}):**\n" + ", ".join(names)
    if cmd_lower == "debug shifts":
        return "**Shift Codes:**\n" + "\n".join(f"• **{k}** — {v}" for k, v in SHIFT_MEANINGS.items())
    if cmd_lower == "debug months":
        tables = get_available_months()
        return f"**Monthly tables ({len(tables)}):**\n" + "\n".join(f"• {t}" for t in tables)
    return None


# ================= UPDATE PIPELINE =================

def parse_update_intent(user_message: str, history: list[dict]) -> dict:
    """Ask LLM to parse the update request into structured JSON."""
    messages = [{"role": "system", "content": SYSTEM_UPDATE}]
    for msg in history[-4:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip any markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        logger.warning(f"Could not parse update JSON: {raw}")
        return {"action": "unclear"}


def find_staff_name(partial: str) -> Optional[str]:
    """Fuzzy-match a partial staff name to the exact DB name."""
    safe = partial.replace("'", "''")
    rows = run_sql(f"SELECT DISTINCT staff_name FROM roster WHERE staff_name ILIKE '%{safe}%' LIMIT 5")
    if rows:
        return rows[0]["staff_name"]
    return None


def execute_update(parsed: dict) -> tuple[bool, str, str]:
    """
    Execute the update on both the main roster table and the monthly table.
    Returns (success, old_shift, message).
    """
    staff_partial = parsed.get("staff_name", "")
    date_str = parsed.get("date", "")
    new_shift = parsed.get("new_shift", "")

    if not staff_partial or not date_str or not new_shift:
        return False, "", "Missing update details — I need a name, date, and shift."

    # Resolve partial name
    staff_name = find_staff_name(staff_partial)
    if not staff_name:
        return False, "", f"I couldn't find anyone matching **{staff_partial}** in the roster."

    # Get old shift
    safe_name = staff_name.replace("'", "''")
    old_rows = run_sql(f"""
        SELECT shift FROM roster
        WHERE staff_name ILIKE '%{safe_name}%'
        AND date = '{date_str}'
        LIMIT 1
    """)
    old_shift = old_rows[0]["shift"] if old_rows else "unknown"

    # Update main roster table
    ok_main = rest_update("roster", staff_name, date_str, new_shift)

    # Update monthly table
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        tbl = month_table(d.year, d.month)
        if table_exists(tbl):
            rest_update(tbl, staff_name, date_str, new_shift)
    except Exception as e:
        logger.warning(f"Monthly table update skipped: {e}")

    if ok_main:
        msg = (
            f"✅ **Updated!**\n\n"
            f"**{staff_name}** on **{format_date(date_str)}**\n"
            f"**{SHIFT_MEANINGS.get(old_shift, old_shift)}** → **{SHIFT_MEANINGS.get(new_shift, new_shift)}**"
        )
        return True, old_shift, msg
    else:
        return False, old_shift, f"❌ Update failed for **{staff_name}** on {format_date(date_str)}. Please check the name and date."


# ================= CORE PIPELINE =================

def text_to_sql(question: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_SQL}]
    for msg in history[-8:]:
        messages.append(msg)
    messages.append({"role": "user", "content": f"Convert to SQL: {question}"})
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, temperature=0.1, max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def sql_to_text(question: str, sql: str, results: list, history: list[dict]) -> str:
    results_str = json.dumps(results[:50], default=str)  # cap at 50 rows for prompt
    user_prompt = (
        f'The user asked: "{question}"\n\n'
        f"SQL used: {sql}\n\n"
        f"Results ({len(results)} rows): {results_str}\n\n"
        f"Please answer the user's question naturally."
    )
    messages = [{"role": "system", "content": SYSTEM_NL}]
    for msg in history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": user_prompt})
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, temperature=0.4, max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def chat_freely(question: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_NL}]
    for msg in history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(
        model=MODEL_NAME, messages=messages, temperature=0.7, max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


# ================= CHAT ENDPOINT =================

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        user_message = ""
        history = []
        for msg in request.messages:
            if msg.role == "user":
                user_message = msg.content
            history.append({"role": msg.role, "content": msg.content})

        if not user_message:
            user_message = "Hello"

        logger.info(f"User: {user_message}")

        # ── Built-in commands ────────────────────────────────────────────────
        builtin = await handle_builtin(user_message)
        if builtin:
            answer = builtin

        # ── Update / change intent ───────────────────────────────────────────
        elif is_update_intent(user_message):
            parsed = parse_update_intent(user_message, history[:-1])
            logger.info(f"Parsed update: {parsed}")

            if parsed.get("action") == "unclear":
                answer = (
                    "I understood you want to make a change, but I need a bit more detail. "
                    "Could you tell me:\n"
                    "- **Who** should be changed?\n"
                    "- **Which date**?\n"
                    "- **What shift** should they be on?\n\n"
                    "Example: *\"Change Sara's shift on March 5th to Day shift\"*"
                )
            else:
                success, old_shift, update_msg = execute_update(parsed)
                answer = update_msg

                # If update succeeded and user wants to see the month roster
                if success and parsed.get("show_month"):
                    try:
                        d = datetime.strptime(parsed["date"], "%Y-%m-%d")
                        tbl = month_table(d.year, d.month)
                        rows = rest_get_month(tbl) if table_exists(tbl) else run_sql(
                            f"SELECT staff_name, staff_id, date, shift FROM roster "
                            f"WHERE date >= '{d.year}-{d.month:02d}-01' "
                            f"AND date <= '{d.year}-{d.month:02d}-{calendar.monthrange(d.year, d.month)[1]}' "
                            f"ORDER BY date, staff_name"
                        )
                        if rows:
                            answer += "\n\n" + format_month_roster(rows, d.year, d.month)
                    except Exception as e:
                        logger.warning(f"Could not fetch month roster after update: {e}")

        # ── Read / query intent ──────────────────────────────────────────────
        elif is_roster_question(user_message):
            raw_sql = text_to_sql(user_message, history[:-1])
            sql = clean_sql(raw_sql)
            logger.info(f"Generated SQL: {sql}")

            if not sql or not sql.upper().startswith("SELECT"):
                sql = "SELECT staff_name, shift, date FROM roster ORDER BY date LIMIT 20"

            results = run_sql(sql)
            logger.info(f"Rows returned: {len(results)}")

            if not results:
                available = get_available_dates()
                hint = f" Data is available from: {', '.join(available[:5])}." if available else ""
                answer = f"I couldn't find any matching records.{hint} Try rephrasing or type `debug dates`."
            else:
                answer = sql_to_text(user_message, sql, results, history[:-1])

        # ── Conversational ───────────────────────────────────────────────────
        else:
            answer = chat_freely(user_message, history[:-1])

        logger.info(f"Answer preview: {answer[:100]}")

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "roster-assistant",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        }

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "roster-assistant",
            "choices": [{"index": 0, "message": {"role": "assistant",
                                                 "content": "Sorry, I ran into an error. Please try again or type `help`."},
                         "finish_reason": "stop"}],
        }


# ================= UTILITY ENDPOINTS =================

@app.get("/")
@app.get("/health")
async def health():
    count = run_sql("SELECT COUNT(*) as cnt FROM roster")
    records = count[0]["cnt"] if count else "unknown"
    months = get_available_months()
    return {
        "status": "healthy",
        "records": records,
        "monthly_tables": len(months),
        "monthly_table_list": months,
        "model": MODEL_NAME,
    }


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "roster-assistant", "object": "model", "created": int(time.time()), "owned_by": "organization"}],
    }


@app.get("/debug/dates")
async def debug_dates():
    total = run_sql("SELECT COUNT(*) as cnt FROM roster")
    rows = run_sql("SELECT DISTINCT date FROM roster ORDER BY date LIMIT 30")
    return {
        "total_records": total[0]["cnt"] if total else 0,
        "dates": [format_date(r["date"]) for r in rows if r.get("date")],
    }


@app.get("/debug/staff")
async def debug_staff():
    rows = run_sql("SELECT DISTINCT staff_name FROM roster ORDER BY staff_name")
    return {"staff": [r["staff_name"] for r in rows if r.get("staff_name")]}


@app.get("/debug/months")
async def debug_months():
    return {"monthly_tables": get_available_months()}

# ================= ENTRYPOINT =================

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run(app, host="0.0.0.0", port=port)