from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    uid: str  # 数据模型
    input: str # 玩家输入
    role_statu: Optional[str] = None  # 角色状态
    chat_situation: Optional[str] = None  # 对话情境

class WriteDiary(BaseModel):
    uid: str  # 数据模型
    date:  Optional[str] = None # 日期
