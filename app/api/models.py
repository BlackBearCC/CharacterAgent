from typing import Optional

from pydantic import BaseModel

class AddGameUser(BaseModel):
    game_uid: str
    username: str = None
    email: str = None

class ChatRequest(BaseModel):
    uid: str  # 用户id
    input: str # 玩家输入
    role_statu: Optional[str] = None  # 角色状态
    chat_situation: Optional[str] = None  # 对话情境

class WriteDiary(BaseModel):
    uid: str  # 用户id
    date:  Optional[str] = None # 日期

class EventRequest(BaseModel):
    uid: str  # 用户id
    need_response:str  # 是否需要回复
    create_at:  Optional[str] = None
    event_location:  Optional[str] = None
    role_statu: Optional[str] = None  # 角色状态
    event_type: str # 事件类型
    event_name: str # 事件
    event_description: Optional[str] = None  # 事件描述
    event_feedback: Optional[str] = None  # 事件反馈
    anticipatory_reaction: Optional[str] = None  # 预期Ai反应
