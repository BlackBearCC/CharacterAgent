from datetime import datetime
from typing import Optional

from dateutil.parser import parse
from pydantic import BaseModel
from pydantic import validator,ValidationError


class GameUser(BaseModel):
    game_uid: str
    user_name: Optional[str] =  None
    role_name : Optional[str] = None
    email: Optional[str] = None


class RoleLog(BaseModel):
    uid: str  # 用户id
    create_at: Optional[str] = None
    log: str

    @validator('create_at')
    def parse_create_at(cls, v):
        try:
            # 使用 dateutil.parser.parse 解析日期时间字符串
            parsed_date = parse(v)
            # 确保解析后的日期时间格式与数据库要求的格式一致
            assert parsed_date.strftime("%Y-%m-%d %H:%M:%S") == v
            return parsed_date
        except (ValueError, AssertionError) as e:
            raise ValidationError("Invalid create_at format.")


class ChatRequest(BaseModel):
    uid: str  # 用户id
    input: str # 玩家输入
    role_status: Optional[str] = ""  # 角色状态
    chat_situation: Optional[str] = ""  # 对话情境

class WriteDiary(BaseModel):
    uid: str  # 用户id
    date_start:  Optional[str] = None # 日期
    date_end: Optional[str] = None  # 日期

class EventRequest(BaseModel):
    uid: str  # 用户id
    need_response:bool = False  # 是否需要回复
    create_at:  Optional[str] = None
    event_location:  Optional[str] = None
    role_status: Optional[str] = None  # 角色状态
    event_type: str # 事件类型
    event_name: str # 事件名
    event_description: Optional[str] = None  # 事件详情资料
    event_feedback: Optional[str] = None  # 事件获得的反馈
    anticipatory_reaction: Optional[str] = None  # 预期Ai反应
