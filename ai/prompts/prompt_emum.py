from enum import Enum


class PromptType(Enum):
    EVENT = "event_response"
    WRITE_DIARY = "write_diary"
    FAST_CHAT = "fast_chat"
    DEEP_CHAT = "deep_chat"
