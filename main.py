import os, time, requests, re, json, calendar, logging
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI

# ── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── ENV ───────────────────────────────────────────────────────────────────────
SUPABASE_URL  = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY  = os.environ.get("SUPABASE_SERVICE_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── OPENAI CLIENT ─────────────────────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL  = "gpt-4o-mini"   # $0.15/M input · $0.60/M output — reliable, fast, cheap

# ── SHIFT CODES ───────────────────────────────────────────────────────────────
SHIFT_MEANINGS = {
    "D":      "Day shift (06:00–15:00)",
    "D*":     "Day shift – modified/extended hours",
    "D-OJT":  "Day shift – On-the-Job Training",
    "D.OT":   "Day shift with Overtime",
    "DP":     "Day Phone – on-call during day hours",
    "A":      "Afternoon shift (14:00–23:00)",
    "A*":     "Afternoon shift – modified/extended hours",
    "AP":     "Afternoon Phone – on-call during afternoon hours",
    "AP/OT":  "Afternoon Phone with Overtime",
    "N":      "Night shift (16:00–22:00)",
    "NP":     "Night Phone – on-call during night hours",
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

# Shifts that count as "absent / not working"
ABSENT_SHIFTS = {"OFF", "V", "PV", "SL", "Sick", "sick", "V-Sick", "Vsick", "BL", "FL", "DIL"}
# Shifts that are actual working shifts
WORKING_SHIFTS = {"D", "D*", "DP", "A", "A*", "AP", "N", "NP", "D-OJT", "D.OT",
                  "OT", "OT-AP", "OT-DP", "OFF-OT", "T", "TRNG", "DT", "PH"}

# ── FASTAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="CAMO Roster Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── SUPABASE HELPERS ──────────────────────────────────────────────────────────
def _h():
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"}

def run_sql(sql: str) -> list:
    try:
        resp = requests.post(f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
                             headers=_h(), json={"query": sql.strip().rstrip(";")}, timeout=30)
        if resp.status_code != 200:
            logger.error(f"SQL error {resp.status_code}: {resp.text[:200]}")
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

def rest_get(table: str, params: dict = None) -> list:
    try:
        p = {"select": "staff_name,staff_id,date,shift", "order": "date,staff_name"}
        if params:
            p.update(params)
        resp = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=_h(), params=p, timeout=30)
        return resp.json() if resp.status_code == 200 else []
    except Exception as e:
        logger.error(f"rest_get: {e}")
        return []

def table_exists(t: str) -> bool:
    r = run_sql(f"SELECT 1 FROM information_schema.tables WHERE table_name='{t}' AND table_schema='public'")
    return bool(r)

def month_table(y: int, m: int) -> str:
    return f"roster_{y:04d}_{m:02d}"

def get_monthly_tables() -> list[str]:
    r = run_sql("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'roster_%_%' AND table_schema='public' ORDER BY table_name")
    return [x["table_name"] for x in r if x.get("table_name")]

def find_staff(partial: str) -> Optional[str]:
    safe = partial.replace("'", "''")
    r = run_sql(f"SELECT DISTINCT staff_name FROM roster WHERE staff_name ILIKE '%{safe}%' LIMIT 5")
    return r[0]["staff_name"] if r else None

def get_shift(staff_name: str, date: str, table: str = "roster") -> Optional[str]:
    safe = staff_name.replace("'", "''")
    r = run_sql(f"SELECT shift FROM {table} WHERE staff_name='{safe}' AND date='{date}' LIMIT 1")
    return r[0]["shift"] if r else None

def fmt_date(d: str) -> str:
    try:
        dt = datetime.strptime(d.split()[0], "%Y-%m-%d")
        day = dt.day
        sfx = "th" if 10 <= day <= 20 else {1:"st",2:"nd",3:"rd"}.get(day%10,"th")
        return dt.strftime(f"%B {day}{sfx}, %Y")
    except:
        return d

# ── SCHEMA ────────────────────────────────────────────────────────────────────
def load_schema():
    try:
        cols = run_sql("SELECT column_name, data_type FROM information_schema.columns WHERE table_name='roster' ORDER BY ordinal_position")
        desc = ", ".join(
            f"{c['column_name']} TEXT('YYYY-MM-DD')" if c.get("column_name") == "date"
            else f"{c.get('column_name')} {c.get('data_type')}" for c in cols
        )
        return f"roster({desc})" if desc else "roster(id, staff_name TEXT, staff_id TEXT, date TEXT, shift TEXT)"
    except:
        return "roster(id, staff_name TEXT, staff_id TEXT, date TEXT, shift TEXT)"

SCHEMA = load_schema()
TODAY  = datetime.now().strftime("%Y-%m-%d")
CUR_Y, CUR_M = datetime.now().year, datetime.now().month

# ── PROMPTS ───────────────────────────────────────────────────────────────────
SHIFTS_JSON = json.dumps(SHIFT_MEANINGS, indent=2)

SYSTEM_INTENT = f"""You are a roster assistant intent classifier. Today is {TODAY}.

Classify the user message into ONE of these intents and return ONLY a JSON object:

1. READ   — user wants to query/view roster data
2. UPDATE — user wants to change/update a shift
3. SUGGEST— user wants replacement suggestions for an absence
4. CHAT   — general conversation, not roster related

Return format:
{{"intent": "READ"|"UPDATE"|"SUGGEST"|"CHAT"}}

Examples:
"Who is working today?" → {{"intent": "READ"}}
"Show March roster"    → {{"intent": "READ"}}
"Change Sara to OFF on March 5" → {{"intent": "UPDATE"}}
"Sara is sick on April 3, who can replace her?" → {{"intent": "SUGGEST"}}
"Hi how are you" → {{"intent": "CHAT"}}
"""

SYSTEM_SQL = f"""You are a PostgreSQL expert. Convert natural language to a single SQL SELECT query.

SCHEMA:
Main table: {SCHEMA}
Monthly tables: roster_YYYY_MM (same columns, one month each) e.g. roster_2026_03

SHIFT CODES:
{SHIFTS_JSON}

DATE RULES — date column is TEXT 'YYYY-MM-DD', today = {TODAY}:
- Today:      date = '{TODAY}'
- Tomorrow:   date = '{(datetime.now()+timedelta(days=1)).strftime("%Y-%m-%d")}'
- This week:  date >= '{TODAY}' AND date <= '{(datetime.now()+timedelta(days=6)).strftime("%Y-%m-%d")}'
- Next month: use table roster_{CUR_Y}_{(CUR_M%12)+1:02d}
- This month: use table roster_{CUR_Y}_{CUR_M:02d}
- Named day:  to_char(date::date,'Day') ILIKE 'Monday%'

TABLE RULES:
- Specific month query → use roster_YYYY_MM table (faster)
- Multi-month or unspecified → use roster
- Use ILIKE for name matching

OUTPUT: Only the SQL query. No markdown, no backticks, no semicolons.
"""

SYSTEM_UPDATE = f"""You are a roster update parser. Today is {TODAY}.

Parse the user message and return ONLY a JSON object:
{{
  "staff_name": "<name as mentioned>",
  "date": "<YYYY-MM-DD>",
  "new_shift": "<shift code from list>",
  "show_month": true/false,
  "reason": "<why if mentioned, else null>"
}}

Valid shift codes: {json.dumps(list(SHIFT_MEANINGS.keys()))}

Mapping hints:
"day off" / "off" → "OFF"
"day shift" / "day" → "D"
"afternoon" → "A"
"afternoon phone" → "AP"
"night" → "N"
"sick" → "Sick"
"vacation" / "leave" → "V"
"training" → "T"

Set show_month=true if user asks to see the updated roster/schedule.
If you can't parse confidently, return {{"error": "unclear", "message": "<what's missing>"}}.
Return ONLY the JSON.
"""

SYSTEM_SUGGEST = f"""You are an expert CAMO roster manager. Today is {TODAY}.

You will be given:
1. The absent staff member's name, date, and their usual shift
2. All other staff schedules for that date AND surrounding days (±2 days for context)
3. The full month's roster data

Your job is to suggest smart, fair replacements. Think like a real scheduler:

RULES:
- Only suggest staff who are working that day (shift != OFF/V/SL etc.)
- Check if suggested staff worked the previous day or will work next day (avoid fatigue)
- Prefer staff on similar shift times (e.g. replace a Day shift with another Day person)
- If no same-shift staff available, suggest a shift swap (e.g. move someone from DP to D)
- Check how many days that person has already worked that week before suggesting overtime
- NEVER suggest someone who is also on leave/off that day

RESPONSE FORMAT:
- Start with a brief summary of the coverage gap
- List 2-3 concrete suggestions with reasoning
- For each suggestion: staff name, what change to make, why they are suitable
- End with a recommended action (which suggestion is best and why)

Be concise, practical, and specific. Use actual names from the data.
"""

SYSTEM_NL = f"""You are a friendly, professional HR assistant for a CAMO aviation department roster.

SHIFT MEANINGS:
{SHIFTS_JSON}

RESPONSE RULES:
- Use full shift names ("Day shift", not "D")
- Format dates nicely ("March 5th, 2026")
- OFF = day off / rest day
- Be warm, concise, professional
- For staff lists use natural language with commas and "and"
- Grammar: "there is 1 person" vs "there are 3 people"
- When showing a change: bold the key info — who, date, old→new shift
- Keep responses focused and scannable
"""

# ── LLM CALLS ─────────────────────────────────────────────────────────────────
def llm(system: str, user: str, history: list = None, temp: float = 0.2, max_tok: int = 600) -> str:
    msgs = [{"role": "system", "content": system}]
    if history:
        msgs.extend(history[-8:])
    msgs.append({"role": "user", "content": user})
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=msgs, temperature=temp, max_tokens=max_tok)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return ""

def parse_json_response(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except:
        return {}

def get_intent(message: str) -> str:
    raw = llm(SYSTEM_INTENT, message, temp=0.0, max_tok=30)
    d = parse_json_response(raw)
    return d.get("intent", "CHAT")

def clean_sql(raw: str) -> str:
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip().rstrip(";").strip()
    m = re.search(r"(SELECT\s.+)", raw, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else raw

# ── READ PIPELINE ─────────────────────────────────────────────────────────────
def handle_read(message: str, history: list) -> str:
    raw_sql = llm(SYSTEM_SQL, f"Convert to SQL: {message}", history, temp=0.1, max_tok=400)
    sql = clean_sql(raw_sql)
    logger.info(f"[SQL] {sql[:200]}")

    if not sql.upper().startswith("SELECT"):
        sql = f"SELECT staff_name, shift, date FROM roster WHERE date='{TODAY}' ORDER BY staff_name"

    rows = run_sql(sql)
    if not rows:
        avail = run_sql("SELECT DISTINCT date FROM roster ORDER BY date LIMIT 5")
        hint = ""
        if avail:
            hint = f" Available dates include: {', '.join(fmt_date(r['date']) for r in avail)}."
        return f"I couldn't find any matching records.{hint} Try rephrasing or type `debug dates`."

    results_str = json.dumps(rows[:60], default=str)
    prompt = (f'User asked: "{message}"\nSQL: {sql}\nResults ({len(rows)} rows): {results_str}\n'
              f'Answer naturally and concisely.')
    return llm(SYSTEM_NL, prompt, history, temp=0.3, max_tok=800)

# ── UPDATE PIPELINE ───────────────────────────────────────────────────────────
def handle_update(message: str, history: list) -> str:
    raw = llm(SYSTEM_UPDATE, message, history, temp=0.0, max_tok=200)
    parsed = parse_json_response(raw)
    logger.info(f"[UPDATE parsed] {parsed}")

    if "error" in parsed:
        return (f"I understood you want to make a change, but I need more info:\n"
                f"{parsed.get('message','')}\n\n"
                f"Example: *\"Change Sara's shift on March 5th to Day shift\"*")

    staff_partial = parsed.get("staff_name", "")
    date_str      = parsed.get("date", "")
    new_shift     = parsed.get("new_shift", "")

    if not all([staff_partial, date_str, new_shift]):
        return ("I need three things to make a change:\n"
                "- **Who** (staff name)\n- **Which date**\n- **New shift**\n\n"
                "Example: *\"Set Muawia to OFF on April 10th\"*")

    staff_name = find_staff(staff_partial)
    if not staff_name:
        return f"I couldn't find anyone matching **{staff_partial}** in the roster. Try `debug staff` to see all names."

    old_shift = get_shift(staff_name, date_str) or "unknown"

    # Update both main table and monthly table
    ok = rest_patch("roster", staff_name, date_str, new_shift)
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        tbl = month_table(d.year, d.month)
        if table_exists(tbl):
            rest_patch(tbl, staff_name, date_str, new_shift)
    except Exception as e:
        logger.warning(f"Monthly table update skipped: {e}")

    if not ok:
        return (f"❌ Update failed for **{staff_name}** on {fmt_date(date_str)}.\n"
                f"Check that the name and date exist in the roster.")

    old_label = SHIFT_MEANINGS.get(old_shift, old_shift)
    new_label = SHIFT_MEANINGS.get(new_shift, new_shift)
    answer = (f"✅ **Shift updated!**\n\n"
              f"**Staff:** {staff_name}\n"
              f"**Date:** {fmt_date(date_str)}\n"
              f"**Change:** {old_label} → **{new_label}**")

    # Optionally show the month roster after update
    if parsed.get("show_month"):
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            tbl = month_table(d.year, d.month)
            rows = rest_get(tbl) if table_exists(tbl) else run_sql(
                f"SELECT staff_name,date,shift FROM roster "
                f"WHERE date>='{d.year}-{d.month:02d}-01' AND date<='{d.year}-{d.month:02d}-{calendar.monthrange(d.year,d.month)[1]}' "
                f"ORDER BY date,staff_name")
            if rows:
                answer += "\n\n" + fmt_month_roster(rows, d.year, d.month)
        except Exception as e:
            logger.warning(f"Post-update month view failed: {e}")

    return answer

# ── SUGGESTION PIPELINE ───────────────────────────────────────────────────────
def handle_suggest(message: str, history: list) -> str:
    # First extract who/when from the message
    extract_prompt = f"""Extract the absent staff member and date from this message.
Today is {TODAY}.
Return JSON only: {{"staff_name": "...", "date": "YYYY-MM-DD"}}
Message: {message}"""
    raw = llm("You extract info from text. Return only JSON.", extract_prompt, temp=0.0, max_tok=100)
    info = parse_json_response(raw)

    staff_partial = info.get("staff_name", "")
    date_str      = info.get("date", "")

    if not staff_partial or not date_str:
        return ("To suggest a replacement I need to know:\n"
                "- **Who** is absent\n- **Which date**\n\n"
                "Example: *\"Sara is sick on March 15th, who can cover her?\"*")

    staff_name = find_staff(staff_partial)
    if not staff_name:
        return f"I couldn't find **{staff_partial}** in the roster. Try `debug staff`."

    # Get the absent person's usual shift on that date
    absent_shift = get_shift(staff_name, date_str) or "D"

    # Get all staff schedules for that date ± 2 days for fatigue context
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        d_minus2 = (dt - timedelta(days=2)).strftime("%Y-%m-%d")
        d_plus2  = (dt + timedelta(days=2)).strftime("%Y-%m-%d")
    except:
        d_minus2 = date_str
        d_plus2  = date_str

    context_rows = run_sql(
        f"SELECT staff_name, date, shift FROM roster "
        f"WHERE date>='{d_minus2}' AND date<='{d_plus2}' "
        f"ORDER BY staff_name, date")

    # Get full month for workload context
    try:
        month_rows = run_sql(
            f"SELECT staff_name, date, shift FROM roster "
            f"WHERE date>='{dt.year}-{dt.month:02d}-01' "
            f"AND date<='{dt.year}-{dt.month:02d}-{calendar.monthrange(dt.year,dt.month)[1]}' "
            f"ORDER BY staff_name, date")
    except:
        month_rows = []

    # Build context string
    context_str  = json.dumps(context_rows,  default=str)
    month_str    = json.dumps(month_rows[:200], default=str)

    user_prompt = (
        f"ABSENT STAFF: {staff_name}\n"
        f"DATE: {fmt_date(date_str)} ({date_str})\n"
        f"THEIR SHIFT: {absent_shift} ({SHIFT_MEANINGS.get(absent_shift, absent_shift)})\n\n"
        f"ALL STAFF SCHEDULES (±2 days around absence date):\n{context_str}\n\n"
        f"FULL MONTH WORKLOAD (for fatigue/fairness check):\n{month_str}\n\n"
        f"Please analyse and suggest the best 2-3 replacement options."
    )

    return llm(SYSTEM_SUGGEST, user_prompt, history, temp=0.4, max_tok=1000)

# ── MONTH ROSTER FORMATTER ────────────────────────────────────────────────────
def fmt_month_roster(rows: list, year: int, month: int) -> str:
    if not rows:
        return "No data."
    month_name = datetime(year, month, 1).strftime("%B %Y")
    staff_list = sorted(set(r["staff_name"] for r in rows))
    days       = sorted(set(r["date"] for r in rows))
    lookup     = {(r["staff_name"], r["date"]): r["shift"] for r in rows}

    lines = [f"**{month_name} Roster**\n"]
    for s in staff_list:
        parts = []
        for d in days:
            shift = lookup.get((s, d), "–")
            day_n = d.split("-")[2].lstrip("0")
            parts.append(f"{day_n}:{shift}")
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

**Make changes:**
- "Change Sara's shift on March 5th to Day shift"
- "Give Muawia a day off on April 10th"
- "Set Gatot to Sick on March 20 and show me March"
- "Update Qadir's shift on the 15th to Afternoon Phone"

**Get replacement suggestions:**
- "Sara is sick on March 15th, who can cover her?"
- "Muawia called in absent on April 3rd, suggest replacements"
- "Geoffrey is off on March 20, who can take his shift?"

**Debug:**
`debug staff` · `debug shifts` · `debug dates` · `debug months` · `help`"""

async def handle_builtin(cmd: str) -> Optional[str]:
    c = cmd.strip().lower()
    if c == "help":
        return HELP_TEXT
    if c == "debug staff":
        r = run_sql("SELECT DISTINCT staff_name FROM roster ORDER BY staff_name")
        names = [x["staff_name"] for x in r if x.get("staff_name")]
        return f"**Staff ({len(names)}):**\n" + ", ".join(names)
    if c == "debug shifts":
        return "**Shift Codes:**\n" + "\n".join(f"• **{k}** — {v}" for k, v in SHIFT_MEANINGS.items())
    if c == "debug dates":
        r = run_sql("SELECT DISTINCT date FROM roster ORDER BY date LIMIT 30")
        return "**Dates in DB:**\n" + "\n".join(f"• {fmt_date(x['date'])}" for x in r if x.get("date"))
    if c == "debug months":
        t = get_monthly_tables()
        return f"**Monthly tables ({len(t)}):**\n" + "\n".join(f"• {x}" for x in t)
    return None

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

        meta_keywords = [
            "### Task:",
            "Suggest 3-5 relevant follow-up questions",
            "Generate a concise, 3-5 word title with an emoji",
            "Generate 1-3 broad tags categorizing",
            "summarizing the chat history"
        ]

        if any(keyword in user_msg for keyword in meta_keywords):
            logger.info(f"Skipping OpenWebUI meta-prompt: {user_msg[:100]}")
            # Return a minimal response that OpenWebUI will ignore
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "roster-assistant",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": ""  # Empty response won't trigger more prompts
                    },
                    "finish_reason": "stop"
                }],
            }

        if not user_msg:
            user_msg = "Hello"
        logger.info(f"User: {user_msg[:100]}")

        # Built-in command check
        builtin = await handle_builtin(user_msg)
        if builtin:
            answer = builtin
        else:
            # Classify intent
            intent = get_intent(user_msg)
            logger.info(f"Intent: {intent}")

            if   intent == "UPDATE":  answer = handle_update(user_msg, history[:-1])
            elif intent == "SUGGEST": answer = handle_suggest(user_msg, history[:-1])
            elif intent == "READ":    answer = handle_read(user_msg, history[:-1])
            else:                     answer = llm(SYSTEM_NL, user_msg, history[:-1], temp=0.7, max_tok=400)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "roster-assistant",
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
        }
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "roster-assistant",
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": "Sorry, something went wrong. Please try again or type `help`."},
                         "finish_reason": "stop"}],
        }

# ── UTILITY ENDPOINTS ─────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
async def health():
    cnt = run_sql("SELECT COUNT(*) AS c FROM roster")
    return {"status": "healthy", "records": cnt[0]["c"] if cnt else "?",
            "model": MODEL, "monthly_tables": len(get_monthly_tables())}

@app.get("/v1/models")
@app.get("/models")
async def models():
    return {"object": "list", "data": [
        {"id": "roster-assistant", "object": "model",
         "created": int(time.time()), "owned_by": "organization"}]}

@app.get("/debug/dates")
async def dbg_dates():
    r = run_sql("SELECT DISTINCT date FROM roster ORDER BY date LIMIT 30")
    return {"dates": [fmt_date(x["date"]) for x in r if x.get("date")]}

@app.get("/debug/staff")
async def dbg_staff():
    r = run_sql("SELECT DISTINCT staff_name FROM roster ORDER BY staff_name")
    return {"staff": [x["staff_name"] for x in r if x.get("staff_name")]}

@app.get("/debug/months")
async def dbg_months():
    return {"monthly_tables": get_monthly_tables()}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))