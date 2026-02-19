import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import OpenAI

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

llm = OpenAI(
    model="gemini-pro",
    openai_api_key=GEMINI_API_KEY,
    openai_api_base="https://generativelanguage.googleapis.com/v1beta"
)

app = FastAPI()

class Question(BaseModel):
    question: str


SYSTEM_PROMPT = """
You are a SQL assistant.

Only answer using database results.

Table: roster(staff_id, staff_name, date, shift)

Shift codes:
D = Day
OFF = Off duty
NP = Night Phone
N = Night
A = Afternoon
AP = Afternoon Phone

For codes you don't know just return codes

Never invent data.
Always query first.
"""


def run_sql(sql):
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/rpc/run_sql",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        },
        json={"query": sql}
    )
    return r.json()


@app.post("/ask")
def ask(q: Question):

    prompt = SYSTEM_PROMPT + "\nQuestion: " + q.question

    sql = llm(prompt)

    result = run_sql(sql)

    return {
        "sql": sql,
        "result": result
    }
