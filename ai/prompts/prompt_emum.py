from enum import Enum


class PromptType(Enum):
    EVENT = "event_response"
    WRITE_DIARY = "write_diary"
    CHAT = "chat"
