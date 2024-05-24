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

class EventRequest(BaseModel):
    uid: str  # 用户id
    event_from: str # 事件发起人
    action: str  # 发起人动作
    action_object: str  # 动作对象
    object_description: Optional[str] = None  # 对象描述
    object_feedback: Optional[str] = None  # 对象反馈
    role_statu: Optional[str] = None  # 角色状态
    anticipatory_reaction: Optional[str] = None  # 预期反应
