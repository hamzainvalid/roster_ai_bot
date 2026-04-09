import os, time, requests, re, json, calendar, logging
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI

# ── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── ENV ───────────────────────────────────────────────────────────────────────
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY   = os.environ.get("SUPABASE_SERVICE_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL  = "gpt-4o-mini"

# ── SHIFT DEFINITIONS ─────────────────────────────────────────────────────────
SHIFT_MEANINGS = {
    "D":      "Day shift (06:00–15:00)",
    "D*":     "Day shift – modified/extended hours",
    "D-OJT":  "Day shift – On-the-Job Training",
    "D.OT":   "Day shift with Overtime",
    "DP":     "Day Phone",
    "A":      "Afternoon shift (14:00–23:00)",
    "A*":     "Afternoon shift – modified/extended hours",
    "AP":     "Afternoon Phone",
    "AP/OT":  "Afternoon Phone with Overtime",
    "N":      "Night shift (16:00–22:00)",
    "NP":     "Night Phone",
    "OFF":    "Day off / Rest day",
    "OFF-OT": "Day off with Overtime called in",
    "OT":     "Overtime",
    "OT-AP":  "Overtime covering Afternoon Phone",
    "OT-DP":  "Overtime covering Day Phone",
    "V":      "Annual Leave (Vacation)",
    "PV":     "Pre-approved Vacation",
    "SL":     "Sick Leave (planned)",
    "Sick":   "Sick (unplanned, same day)",
    "sick":   "Sick (unplanned, same day)",
    "V-Sick": "Vacation converted to Sick Leave",
    "Vsick":  "Vacation converted to Sick Leave",
    "BL":     "Bonus Leave",
    "FL":     "Family Leave",
    "DIL":    "Day in Lieu",
    "T":      "Training",
    "TRNG":   "Training",
    "DT":     "Duty Travel",
    "PH":     "Public Holiday",
}

# Shifts that mean the person is NOT physically present / unavailable
ABSENT_SHIFTS = {
    "OFF", "V", "PV", "SL", "Sick", "sick", "V-Sick", "Vsick",
    "BL", "FL", "DIL", "DT", "PH"
}

# Shifts grouped by time-of-day (for smart replacement matching)
DAY_SHIFTS       = {"D", "D*", "D-OJT", "D.OT", "DP", "OT-DP", "T", "TRNG"}
AFTERNOON_SHIFTS = {"A", "A*", "AP", "AP/OT", "OT-AP"}
NIGHT_SHIFTS     = {"N", "NP"}

SHIFTS_JSON = json.dumps(SHIFT_MEANINGS, indent=2)

# ── FASTAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="CAMO Roster Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── DATE / TABLE HELPERS ──────────────────────────────────────────────────────
def month_table(y: int, m: int) -> str:
    return f"roster_{y:04d}_{m:02d}"

def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def fmt_date(d: str) -> str:
    try:
        dt = datetime.strptime(str(d).split()[0], "%Y-%m-%d")
        day = dt.day
        sfx = "th" if 10 <= day <= 20 else {1:"st",2:"nd",3:"rd"}.get(day % 10, "th")
        return dt.strftime(f"%B {day}{sfx}, %Y")
    except Exception:
        return str(d)

def resolve_table_for_date(date_str: str) -> str:
    """Return the correct monthly table name for a given YYYY-MM-DD date."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return month_table(dt.year, dt.month)
    except Exception:
        now = datetime.now()
        return month_table(now.year, now.month)

def infer_table_from_message(message: str) -> str:
    """
    Best-effort: parse a month name or year-month from free text
    so the SQL generator always gets the right table name.
    Returns e.g. 'roster_2026_04'.
    """
    now = datetime.now()
    msg = message.lower()

    # "next month"
    if "next month" in msg:
        nm = now.month % 12 + 1
        ny = now.year if now.month < 12 else now.year + 1
        return month_table(ny, nm)

    # "last month" / "previous month"
    if "last month" in msg or "previous month" in msg:
        pm = (now.month - 2) % 12 + 1
        py = now.year if now.month > 1 else now.year - 1
        return month_table(py, pm)

    # "this month" or no month hint → current
    if "this month" in msg:
        return month_table(now.year, now.month)

    # Explicit month name e.g. "March 2026", "march", "apr"
    months = {
        "january":1,"jan":1,"february":2,"feb":2,"march":3,"mar":3,
        "april":4,"apr":4,"may":5,"june":6,"jun":6,"july":7,"jul":7,
        "august":8,"aug":8,"september":9,"sep":9,"sept":9,
        "october":10,"oct":10,"november":11,"nov":11,"december":12,"dec":12,
    }
    found_month = None
    found_year  = now.year

    # look for 4-digit year
    yr_match = re.search(r'\b(202[0-9])\b', msg)
    if yr_match:
        found_year = int(yr_match.group(1))

    for name, num in months.items():
        if name in msg:
            found_month = num
            break

    if found_month:
        return month_table(found_year, found_month)

    # Fallback: today's month
    return month_table(now.year, now.month)

# ── SUPABASE HELPERS ──────────────────────────────────────────────────────────
def _h():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }

def run_sql(sql: str) -> list:
    try:
        clean = sql.strip().rstrip(";")
        logger.info(f"[SQL] {clean[:300]}")
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
            headers=_h(), json={"query": clean}, timeout=30)
        if resp.status_code != 200:
            logger.error(f"SQL {resp.status_code}: {resp.text[:200]}")
            return []
        r = resp.json()
        return r if isinstance(r, list) else []
    except Exception as e:
        logger.error(f"run_sql: {e}")
        return []

def rest_patch(table: str, staff_name: str, date: str, new_shift: str) -> bool:
    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={**_h(), "Prefer": "return=minimal"},
            params={"staff_name": f"eq.{staff_name}", "date": f"eq.{date}"},
            json={"shift": new_shift}, timeout=30)
        return resp.status_code in (200, 204)
    except Exception as e:
        logger.error(f"rest_patch: {e}")
        return False

def table_exists(t: str) -> bool:
    r = run_sql(
        f"SELECT 1 FROM information_schema.tables "
        f"WHERE table_name='{t}' AND table_schema='public'")
    return bool(r)

def get_monthly_tables() -> list:
    r = run_sql(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name LIKE 'roster_%_%' AND table_schema='public' "
        "ORDER BY table_name")
    return [x["table_name"] for x in r if x.get("table_name")]

def find_staff_in_table(partial: str, table: str) -> Optional[str]:
    """Find exact staff name via ILIKE in a specific monthly table."""
    safe = partial.replace("'", "''")
    r = run_sql(
        f"SELECT DISTINCT staff_name FROM {table} "
        f"WHERE staff_name ILIKE '%{safe}%' LIMIT 5")
    return r[0]["staff_name"] if r else None

def get_all_staff_on_date(table: str, date_str: str) -> list:
    """Return all rows for every staff member on a specific date."""
    return run_sql(
        f"SELECT staff_name, shift FROM {table} "
        f"WHERE date='{date_str}' ORDER BY staff_name")

def get_full_month(table: str) -> list:
    """Fetch the entire monthly table sorted by date then name."""
    return run_sql(
        f"SELECT staff_name, date, shift FROM {table} "
        f"ORDER BY date, staff_name")

# ── LLM HELPER ────────────────────────────────────────────────────────────────
def llm(system: str, user: str, history: list = None,
        temp: float = 0.2, max_tok: int = 600) -> str:
    msgs = [{"role": "system", "content": system}]
    if history:
        msgs.extend(history[-8:])
    msgs.append({"role": "user", "content": user})
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=msgs,
            temperature=temp, max_completion_tokens=max_tok)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM: {e}")
        return ""

def parse_json(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}

# ── PROMPTS ───────────────────────────────────────────────────────────────────
def make_intent_prompt() -> str:
    today = today_str()
    return f"""You are a roster assistant intent classifier. Today is {today}.

Classify the user message into ONE of:
  READ    — user wants to query/view roster information
  SUGGEST — user asks whether someone CAN take off, or wants replacement suggestions
  CHAT    — general conversation unrelated to the roster

Return ONLY a JSON object, nothing else.
Format: {{"intent": "READ"|"SUGGEST"|"CHAT"}}

Examples:
"Who is working today?"                          → {{"intent": "READ"}}
"Show me the March 2026 roster"                  → {{"intent": "READ"}}
"What shift is Sara on next Monday?"             → {{"intent": "READ"}}
"Can Qadir take off on March 3?"                 → {{"intent": "SUGGEST"}}
"Sara is sick on April 3, who can replace her?"  → {{"intent": "SUGGEST"}}
"Is it possible for Muawia to take off Friday?"  → {{"intent": "SUGGEST"}}
"Hi, how are you?"                               → {{"intent": "CHAT"}}
"""

def make_sql_prompt(table: str) -> str:
    today = today_str()
    now   = datetime.now()
    tom   = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    wend  = (now + timedelta(days=6)).strftime("%Y-%m-%d")
    return f"""You are a PostgreSQL expert. Convert natural language to a single SQL SELECT query.

TARGET TABLE: {table}
COLUMNS: staff_name TEXT, staff_id TEXT, date TEXT (YYYY-MM-DD), shift TEXT

SHIFT CODES:
{SHIFTS_JSON}

DATE RULES (date is stored as TEXT 'YYYY-MM-DD'):
- "today"     → date = '{today}'
- "tomorrow"  → date = '{tom}'
- "this week" → date >= '{today}' AND date <= '{wend}'
- Named day   → to_char(date::date, 'Day') ILIKE 'Monday%'
- A range     → date >= 'YYYY-MM-DD' AND date <= 'YYYY-MM-DD'

NAME MATCHING — ALWAYS use ILIKE with wildcards:
  ✓ staff_name ILIKE '%qadir%'
  ✓ staff_name ILIKE '%qadir%' OR staff_name ILIKE '%marlon%'
  ✗ staff_name = 'Qadir'           (exact match — wrong)
  ✗ staff_name ILIKE ANY(ARRAY[…]) (doesn't work — wrong)

SHIFT MATCHING:
  ✓ shift = 'D'
  ✓ shift NOT IN ('OFF','V','PV','SL','Sick','sick','BL','FL','DIL','DT','PH')
  "working" / "on duty" / "duty" = shift NOT IN (absent list above)
  "off" / "absent" / "off duty"     = shift IN (absent list above)

OUTPUT: Only the raw SQL query. No markdown, no backticks, no semicolons, no explanation.
"""

def make_suggest_prompt() -> str:
    today = today_str()
    return f"""You are an expert CAMO roster manager. Today is {today}.

You will receive:
  1. The requesting staff member's name, the date they want off, and their current shift
  2. A full snapshot of every staff member's shift on that exact date

Apply these rules strictly:

RULE 1 — CAN THEY TAKE OFF?
  • Count how many OTHER staff members are on the SAME shift (or equivalent same-time shift group).
  • Same-time groups:
      Day group      : D, D*, DP, D-OJT, D.OT, OT-DP
      Afternoon group: A, A*, AP, AP/OT, OT-AP
      Night group    : N, NP
  • If ≥ 2 other person is in the same group → They need to contact their manager.
    State: "[list colleagues] are also on [shift group] that day. Contact your manager"
  • If 1 other in same group → They still need to contact their manager.
    State: "[colleague] is also on [shift group] that day. Contact your manager"
  • If 0 others in same group → Just say leave not possible.
    State: "No. Contact your manager"

RULE 2 — SUGGEST REPLACEMENTS (only if needed or explicitly asked)
  • Only consider staff who are WORKING that day (shift NOT in: OFF, V, PV, SL, Sick, sick, BL, FL, DIL, DT, PH).
  • NEVER suggest someone who is on leave/off/absent that day.
  • Prefer staff already in the same shift group — least disruption.
  • If no same-group staff available, look for staff in a shift with MORE THAN ONE person.
    Suggest moving one of them to cover, ensuring their original shift still has ≥ 1 person.
  • List each suggestion as: "[Name] (currently on [shift]) → could cover [target shift]. Rationale: ..."
  • Give at most 3 suggestions, ranked best-first.
  • End with a clear RECOMMENDED ACTION.

RULE 3 — NEVER invent data. Only use names and shifts from the snapshot provided.

FORMAT:
  • Keep it concise and scannable.
  • Use **bold** for names and shift codes.
  • End with a one-sentence recommendation.
"""

def make_nl_prompt() -> str:
    return f"""You are a friendly, professional HR assistant for a CAMO aviation department.

SHIFT MEANINGS:
{SHIFTS_JSON}

STRICT RULES:
  • Only use information explicitly present in the database results provided.
  • NEVER invent or guess shifts, names, or dates.
  • If no results were found, say "I don't have that information in the roster."
  • Do not use full shift names (not "Day shift", "D").
  • Format dates as "March 5th, 2026" (not 2026-03-05).
  • OFF / V / SL etc. = the person is not working / is absent / off duty.
  • Be warm, concise, and professional.
  • Use natural lists: "Alice, Bob, and Carol" not bullet points for short lists.
  • Grammar: "there is 1 person" vs "there are 3 people".
  . Don't use full name just first names.
"""

# ── INTENT ────────────────────────────────────────────────────────────────────
def get_intent(message: str) -> str:
    raw = llm(make_intent_prompt(), message, temp=0.0, max_tok=30)
    return parse_json(raw).get("intent", "CHAT")

# ── READ PIPELINE ─────────────────────────────────────────────────────────────
def clean_sql(raw: str) -> str:
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip().rstrip(";").strip()
    m = re.search(r"(SELECT\s.+)", raw, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else raw

def handle_read(message: str, history: list) -> str:
    # Determine the right monthly table from the message
    table = infer_table_from_message(message)

    # Verify the table exists; if not, tell the user
    if not table_exists(table):
        available = get_monthly_tables()
        avail_str = ", ".join(available[:6]) + ("..." if len(available) > 6 else "")
        return (f"I don't have data for that period. "
                f"Available months: {avail_str}\n"
                f"Type `debug months` to see all available tables.")

    raw_sql = llm(make_sql_prompt(table),
                  f"Convert to SQL: {message}",
                  history, temp=0.1, max_tok=400)
    sql = clean_sql(raw_sql)
    logger.info(f"[READ SQL] {sql[:300]}")

    # Safety fallback
    if not sql.upper().startswith("SELECT"):
        today = today_str()
        sql = f"SELECT staff_name, shift, date FROM {table} WHERE date='{today}' ORDER BY staff_name"

    rows = run_sql(sql)

    if not rows:
        return (f"No matching records found in **{table}** for that query. "
                f"Try rephrasing or check `debug dates`.")

    results_str = json.dumps(rows[:80], default=str)
    prompt = (f'User asked: "{message}"\n'
              f'Table queried: {table}\n'
              f'SQL used: {sql}\n'
              f'Results ({len(rows)} rows):\n{results_str}\n\n'
              f'Answer the question naturally and concisely based ONLY on these results.')
    return llm(make_nl_prompt(), prompt, history, temp=0.3, max_tok=800)

# ── SUGGEST PIPELINE ──────────────────────────────────────────────────────────
def handle_suggest(message: str, history: list) -> str:
    # Step 1: extract staff name + date from the message
    extract_sys  = "Extract information from text. Return ONLY a JSON object."
    today        = today_str()
    extract_user = (
        f"Today is {today}.\n"
        f"Extract the staff member name and the date from the message below.\n"
        f'Return: {{"staff_name": "...", "date": "YYYY-MM-DD"}}\n'
        f"If the date is relative (e.g. 'next Monday', 'this Friday'), resolve it to YYYY-MM-DD.\n"
        f"Message: {message}"
    )
    raw  = llm(extract_sys, extract_user, temp=0.0, max_tok=100)
    info = parse_json(raw)

    staff_partial = info.get("staff_name", "").strip()
    date_str      = info.get("date", "").strip()

    if not staff_partial or not date_str:
        return ("I need two things to check coverage:\n"
                "- **Who** wants to take off or needs replacing\n"
                "- **Which date**\n\n"
                "Example: *\"Can Sara take off on March 15th?\"*")

    # Step 2: resolve the correct monthly table
    table = resolve_table_for_date(date_str)
    if not table_exists(table):
        available = get_monthly_tables()
        avail_str = ", ".join(available[:6])
        return (f"I don't have roster data for that date ({fmt_date(date_str)}). "
                f"Available months: {avail_str}")

    # Step 3: find exact staff name in that table
    staff_name = find_staff_in_table(staff_partial, table)
    if not staff_name:
        return (f"I couldn't find **{staff_partial}** in the "
                f"{datetime.strptime(date_str,'%Y-%m-%d').strftime('%B %Y')} roster.\n"
                f"Type `debug staff` to see all staff names.")

    # Step 4: get that person's shift on the requested date
    shift_rows = run_sql(
        f"SELECT shift FROM {table} "
        f"WHERE staff_name='{staff_name.replace(chr(39), chr(39)+chr(39))}' "
        f"AND date='{date_str}' LIMIT 1")

    if not shift_rows:
        return (f"I couldn't find a roster entry for **{staff_name}** "
                f"on **{fmt_date(date_str)}** in {table}.")

    person_shift = shift_rows[0]["shift"]

    # Step 5: get ALL staff shifts on that same date (the daily snapshot)
    daily_snapshot = get_all_staff_on_date(table, date_str)
    if not daily_snapshot:
        return f"No roster data found for {fmt_date(date_str)} in {table}."

    # Step 6: Build a rich context and call the suggestion LLM
    snapshot_str = json.dumps(daily_snapshot, default=str, indent=2)

    user_prompt = (
        f"REQUESTING STAFF : {staff_name}\n"
        f"DATE             : {fmt_date(date_str)} ({date_str})\n"
        f"THEIR SHIFT      : {person_shift} "
        f"({SHIFT_MEANINGS.get(person_shift, person_shift)})\n"
        f"USER'S QUESTION  : {message}\n\n"
        f"FULL DAILY ROSTER SNAPSHOT (all staff on {date_str}):\n"
        f"{snapshot_str}\n\n"
        f"Apply the rules and provide your assessment."
    )

    return llm(make_suggest_prompt(), user_prompt, history, temp=0.3, max_tok=1000)

# ── UPDATE PIPELINE ───────────────────────────────────────────────────────────
def make_update_prompt() -> str:
    today = today_str()
    codes = json.dumps(list(SHIFT_MEANINGS.keys()))
    return f"""You are a roster update parser. Today is {today}.

Parse the user message and return ONLY a JSON object:
{{
  "staff_name": "<name as mentioned>",
  "date": "<YYYY-MM-DD>",
  "new_shift": "<valid shift code>",
  "show_month": true/false,
  "reason": "<reason if mentioned, else null>"
}}

Valid shift codes: {codes}

Common mappings:
  "day off" / "off"     → "OFF"
  "day shift" / "day"   → "D"
  "afternoon"           → "A"
  "afternoon phone"     → "AP"
  "night"               → "N"
  "sick"                → "Sick"
  "vacation" / "leave"  → "V"
  "training"            → "T"

Set show_month=true if user also wants to see the updated monthly roster.
If you cannot parse all three required fields, return:
  {{"error": "unclear", "message": "<what is missing>"}}

Return ONLY the JSON object.
"""

def handle_update(message: str, history: list) -> str:
    raw    = llm(make_update_prompt(), message, history, temp=0.0, max_tok=200)
    parsed = parse_json(raw)
    logger.info(f"[UPDATE] {parsed}")

    if "error" in parsed:
        return (f"I understood you want to make a change, but I need more info:\n"
                f"{parsed.get('message', '')}\n\n"
                f"Example: *\"Change Sara's shift on March 5th to Day shift\"*")

    staff_partial = parsed.get("staff_name", "")
    date_str      = parsed.get("date", "")
    new_shift     = parsed.get("new_shift", "")

    if not all([staff_partial, date_str, new_shift]):
        return ("I need three things to make a change:\n"
                "- **Who** (staff name)\n- **Which date**\n- **New shift**\n\n"
                "Example: *\"Set Muawia to OFF on April 10th\"*")

    # Resolve table
    table = resolve_table_for_date(date_str)
    if not table_exists(table):
        return f"No roster table found for that date ({fmt_date(date_str)})."

    # Exact name lookup
    staff_name = find_staff_in_table(staff_partial, table)
    if not staff_name:
        return (f"I couldn't find **{staff_partial}** in the roster. "
                f"Type `debug staff` to see all staff names.")

    # Fetch old shift
    old_rows  = run_sql(
        f"SELECT shift FROM {table} "
        f"WHERE staff_name='{staff_name.replace(chr(39),chr(39)*2)}' "
        f"AND date='{date_str}' LIMIT 1")
    old_shift = old_rows[0]["shift"] if old_rows else "unknown"

    # Patch both the monthly table AND the main roster table
    ok = rest_patch(table, staff_name, date_str, new_shift)
    rest_patch("roster", staff_name, date_str, new_shift)  # keep in sync

    if not ok:
        return (f"❌ Update failed for **{staff_name}** on {fmt_date(date_str)}.\n"
                f"Please check the name and date exist in the roster.")

    old_label = SHIFT_MEANINGS.get(old_shift, old_shift)
    new_label = SHIFT_MEANINGS.get(new_shift, new_shift)
    answer = (f"✅ **Shift updated!**\n\n"
              f"**Staff :** {staff_name}\n"
              f"**Date  :** {fmt_date(date_str)}\n"
              f"**Change:** {old_label} → **{new_label}**")

    # Optionally append the month roster
    if parsed.get("show_month"):
        try:
            rows = get_full_month(table)
            dt   = datetime.strptime(date_str, "%Y-%m-%d")
            if rows:
                answer += "\n\n" + fmt_month_roster(rows, dt.year, dt.month)
        except Exception as e:
            logger.warning(f"Month view after update failed: {e}")

    return answer

# ── MONTH ROSTER FORMATTER ────────────────────────────────────────────────────
def fmt_month_roster(rows: list, year: int, month: int) -> str:
    if not rows:
        return "No data."
    label      = datetime(year, month, 1).strftime("%B %Y")
    staff_list = sorted({r["staff_name"] for r in rows})
    days       = sorted({r["date"] for r in rows})
    lookup     = {(r["staff_name"], r["date"]): r["shift"] for r in rows}

    lines = [f"**{label} Roster**\n"]
    for s in staff_list:
        parts = [f"{d.split('-')[2].lstrip('0')}:{lookup.get((s,d),'–')}" for d in days]
        lines.append(f"**{s}**: " + "  ".join(parts))
    return "\n".join(lines)

# ── BUILT-IN COMMANDS ─────────────────────────────────────────────────────────
HELP_TEXT = """**CAMO Roster Assistant — Help**

**View the roster:**
- "Who is working today?"
- "Show me the full March 2026 roster"
- "Who is on Day shift tomorrow?"
- "What is Sara's schedule next week?"
- "How many people are off this Friday?"
- "Show me Qadir, Marlon and Euphracia's shifts for April"

**Check coverage / replacements:**
- "Can Sara take off on March 15th?"
- "Is it possible for Muawia to take off this Friday?"
- "Sara is sick on April 3rd — who can cover her?"
- "Geoffrey called in absent on March 20, suggest replacements"

**Make changes:**
- "Change Sara's shift on March 5th to Day shift"
- "Give Muawia a day off on April 10th"
- "Set Gatot to Sick on March 20 and show me the March roster"
- "Update Qadir's shift on the 15th to Afternoon Phone"

**Debug:**
`debug staff` · `debug shifts` · `debug dates` · `debug months` · `help`"""

async def handle_builtin(cmd: str) -> Optional[str]:
    c = cmd.strip().lower()
    if c == "help":
        return HELP_TEXT
    if c == "debug staff":
        # Pull from any monthly table that exists
        tables = get_monthly_tables()
        tbl    = tables[0] if tables else "roster"
        r      = run_sql(f"SELECT DISTINCT staff_name FROM {tbl} ORDER BY staff_name")
        names  = [x["staff_name"] for x in r if x.get("staff_name")]
        return f"**Staff ({len(names)}):**\n" + ", ".join(names)
    if c == "debug shifts":
        lines = [f"• **{k}** — {v}" for k, v in SHIFT_MEANINGS.items()]
        return "**Shift Codes:**\n" + "\n".join(lines)
    if c == "debug dates":
        tables = get_monthly_tables()
        if not tables:
            return "No monthly tables found."
        tbl = tables[-1]  # most recent
        r   = run_sql(f"SELECT DISTINCT date FROM {tbl} ORDER BY date LIMIT 35")
        return f"**Dates in {tbl}:**\n" + "\n".join(
            f"• {fmt_date(x['date'])}" for x in r if x.get("date"))
    if c == "debug months":
        t = get_monthly_tables()
        return f"**Monthly tables ({len(t)}):**\n" + "\n".join(f"• {x}" for x in t)
    return None

# ── META-PROMPT FILTER (OpenWebUI internal prompts) ───────────────────────────
META_KEYWORDS = [
    "### Task:", "Suggest 3-5 relevant follow-up questions",
    "Generate a concise, 3-5 word title",
    "Generate 1-3 broad tags", "summarizing the chat history",
]

def is_meta_prompt(msg: str) -> bool:
    return any(kw in msg for kw in META_KEYWORDS)

# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "roster-assistant"
    messages: List[Message]
    temperature: Optional[float] = 0.3

# ── CHAT ENDPOINT ─────────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat(request: ChatRequest):
    try:
        user_msg = ""
        history  = []
        for m in request.messages:
            if m.role == "user":
                user_msg = m.content
            history.append({"role": m.role, "content": m.content})

        # Silently drop OpenWebUI internal meta-prompts
        if is_meta_prompt(user_msg):
            return _empty_response()

        if not user_msg.strip():
            user_msg = "Hello"

        logger.info(f"User: {user_msg[:120]}")

        # 1. Built-in commands
        builtin = await handle_builtin(user_msg)
        if builtin:
            return _response(builtin)

        # 2. Check for update intent FIRST (before LLM classification)
        #    to avoid misclassifying "change X to Y" as READ
        update_triggers = [
            "change", "update", "modify", "swap", "switch",
            "set ", "put ", "assign", "replace", "edit",
            "reschedule", "give him", "give her", "make him", "make her",
        ]
        is_update = any(t in user_msg.lower() for t in update_triggers)

        if is_update:
            answer = handle_update(user_msg, history[:-1])
        else:
            intent = get_intent(user_msg)
            logger.info(f"Intent: {intent}")
            if   intent == "SUGGEST": answer = handle_suggest(user_msg, history[:-1])
            elif intent == "READ":    answer = handle_read(user_msg, history[:-1])
            else:                     answer = llm(make_nl_prompt(), user_msg, history[:-1],
                                                   temp=0.7, max_tok=400)

        return _response(answer)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return _response("Sorry, something went wrong. Please try again or type `help`.")

# ── RESPONSE HELPERS ──────────────────────────────────────────────────────────
def _response(content: str) -> dict:
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "roster-assistant",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }

def _empty_response() -> dict:
    return _response("")

# ── UTILITY ENDPOINTS ─────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
async def health():
    monthly = get_monthly_tables()
    return {
        "status":         "healthy",
        "model":          MODEL,
        "monthly_tables": len(monthly),
        "table_list":     monthly,
    }

@app.get("/v1/models")
@app.get("/models")
async def models_list():
    return {"object": "list", "data": [
        {"id": "roster-assistant", "object": "model",
         "created": int(time.time()), "owned_by": "organization"}]}

@app.get("/debug/staff")
async def dbg_staff():
    tables = get_monthly_tables()
    tbl    = tables[0] if tables else "roster"
    r      = run_sql(f"SELECT DISTINCT staff_name FROM {tbl} ORDER BY staff_name")
    return {"staff": [x["staff_name"] for x in r if x.get("staff_name")]}

@app.get("/debug/dates")
async def dbg_dates():
    tables = get_monthly_tables()
    if not tables:
        return {"dates": []}
    tbl = tables[-1]
    r   = run_sql(f"SELECT DISTINCT date FROM {tbl} ORDER BY date LIMIT 35")
    return {"table": tbl, "dates": [fmt_date(x["date"]) for x in r if x.get("date")]}

@app.get("/debug/months")
async def dbg_months():
    return {"monthly_tables": get_monthly_tables()}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))