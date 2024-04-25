# main.py
import asyncio

from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain_community.llms.tongyi import Tongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from ai.models import QianwenModel
from app.core import CharacterAgent

app = FastAPI()


model = QianwenModel()

tuji_agent = CharacterAgent("兔叽",model)
prompt_text = "你好"


async def process_response(text, session_id, query):
    pass
async def main():
    await tuji_agent.response_stream(prompt_text)

asyncio.run(main())
@app.get("/")
def read_root():
    return {"Hello": "World"}
