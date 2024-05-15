from typing import List
from langchain.schema import BaseMessage

def get_prefixed_buffer_string(messages: List[BaseMessage], human_prefix: str, ai_prefix: str) -> str:
    buffer = ""
    for msg in messages:
        if msg.type == "human":
            buffer += f"{human_prefix}: {msg.content}\n"
        elif msg.type == "ai":
            buffer += f"{ai_prefix}: {msg.content}\n"
        else:
            buffer += f"{msg.content}\n"
    return buffer.strip()