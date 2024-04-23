# main.py
import asyncio

from fastapi import FastAPI

from ai.generator import Generator
from ai.models import QianwenModel
from app.core import CharacterAgent

app = FastAPI()

model = QianwenModel()
tuji_agent = CharacterAgent("兔叽",model)


async def process_response(text, session_id, query):

    pass
async def main():
    prompt_text = "你是谁，说中文"
    async for response in tuji_agent.response(prompt_text):
        print(response)


# Run the main coroutine
asyncio.run(main())
@app.get("/")
def read_root():
    return {"Hello": "World"}
