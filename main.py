import os
from fastapi import FastAPI
from pydantic import BaseModel
import requests
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import OpenAI

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GEMINI_API_KEY = os.environ["GEM_API"]

llm = OpenAI(
    model="gemini-2.5-flash",
    openai_api_key=GEMINI_API_KEY,
    openai_api_base="https://generativelanguage.googleapis.com/v1beta"
)

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

app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):

    sql = chain.invoke({"question": q.question})

    result = db.run(sql)

    return {
        "sql": sql,
        "result": result
    }