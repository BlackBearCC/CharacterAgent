
import json
import logging

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from dateutil.parser import parse
from sqlalchemy import Column, Integer, Text, create_engine, text, DateTime, String, ForeignKey
from datetime import datetime
from ai.models.ai import AIMessage
from ai.models.buffer import get_prefixed_buffer_string
from ai.models.human import HumanMessage
from ai.models.system import SystemMessage
from ai.models.user import User

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    messages_from_dict
)
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class BaseMessageConverter(ABC):
    """Convert BaseMessage to the SQLAlchemy model."""

    @abstractmethod
    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        """Convert a SQLAlchemy model to a BaseMessage instance."""
        raise NotImplementedError

    @abstractmethod
    def to_sql_model(self, message: BaseMessage, guid: str) -> Any:
        """Convert a BaseMessage instance to a SQLAlchemy model."""
        raise NotImplementedError

    @abstractmethod
    def get_sql_model_class(self) -> Any:
        """Get the SQLAlchemy model class."""
        raise NotImplementedError


def create_message_model(table_name: str, DynamicBase: Any) -> Any:
    class Message(DynamicBase):  # type: ignore[valid-type, misc]
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        user_guid = Column(String(128), ForeignKey(User.guid))  # 修改为VARCHAR(128)并添加外键约束
        message = Column(Text)  # Set collation to support Unicode characters
        created_at = Column(DateTime, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
        generate_from = Column(Text)
        call_step = Column(Text)

    return Message


class DefaultMessageConverter(BaseMessageConverter):
    """The default message converter for SQLChatMessageHistory."""

    def __init__(self, table_name: str):
        self.model_class = create_message_model(table_name, declarative_base())
        # with self.engine.connect() as connection:
        #     connection.execute(
        #         text(f"ALTER TABLE {table_name} CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"))

    def message_to_dict(self,message: BaseMessage) -> dict:
        """修改 langchain的message_to_dict 方法，替换存入数据库的键值
        """
        return {"type": message.type, "data": message.content}
    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        message_dict = json.loads(sql_message.message)
        # print(message_dict)


        message_type = message_dict.get("type")
        content = message_dict.get("data", {})
        created_at = sql_message.created_at.strftime("%Y-%m-%d %H:%M:%S")  # Convert datetime to string

        if message_type == "human":
            message = HumanMessage(content=content,created_at=created_at)
        elif message_type == "ai":
            message = AIMessage(content=content,created_at=created_at)
        elif message_type == "system":
            message = SystemMessage(content=content,created_at=created_at)
        else:
            message = BaseMessage(content=content)

        # Add the created_at attribute to the message dict
        # message_dict["data"]["created_at"] = created_at
        # message.message = json.dumps(message_dict, ensure_ascii=False)

        return message


    def to_sql_model(self, message: BaseMessage, guid: str) -> Any:
        now = datetime.now()  # 获取当前日期和时间
        # 检查message对象是否有created_at属性
        if hasattr(message, 'created_at') and message.created_at is not None:
            try:
                message_created_at = parse(message.created_at)  # 将字符串转换为datetime
            except ValueError:
                message_created_at = now  # 转换失败，使用当前时间
        else:
            message_created_at = now

        return self.model_class(
            user_guid=guid,
            message=json.dumps(self.message_to_dict(message), ensure_ascii=False),
            generate_from=message.generate_from,
            call_step=message.call_step,
            created_at=message_created_at,
        )
    def get_sql_model_class(self) -> Any:
        return self.model_class


class SQLChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in an SQL database."""

    def __init__(
        self,

        connection_string: str,
        table_name: str = "message_store",
        user_guid_field_name: str = "user_guid",
        custom_message_converter: Optional[BaseMessageConverter] = None,
        _create_table_if_not_exists: bool = True,

    ):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, echo=False)
        self.user_guid_field_name = user_guid_field_name
        self.converter = DefaultMessageConverter(table_name)
        self.sql_model_class = self.converter.get_sql_model_class()
        if not hasattr(self.sql_model_class, user_guid_field_name):
            raise ValueError("SQL model class must have session_id column")
        self._create_table_if_not_exists()


        self.Session = sessionmaker(self.engine)


    def _create_table_if_not_exists(self) -> None:
        self.sql_model_class.metadata.create_all(self.engine)


    def messages(self,guid:str, count: int = 100) -> List[BaseMessage]:
        """
        从数据库中检索指定数量的消息记录。

        参数:
        count (int): 要检索的消息记录数量，默认为100。

        返回值:
        List[BaseMessage]: 检索到的消息记录列表，每个记录都是一个BaseMessage类型的实例。
        """


        with self.Session() as session:
            result = (
                session.query(self.sql_model_class)
                .where(getattr(self.sql_model_class, self.user_guid_field_name) == guid)
                .order_by(self.sql_model_class.id.desc())
                .limit(count)
                .all()  # 添加.all()来执行查询
            )

            messages = []
            for record in reversed(result):
                messages.append(self.converter.from_sql_model(record))

            return messages

    def buffer(self, guid:str,user_name="主人",role_name="兔兔",count: int = 100, with_timestamps: bool = True) -> str:

            _messages = self.messages(guid,count)
            history_buffer = ""

            for message in _messages:
                # timestamp = datetime.datetime.fromtimestamp(message.created_at).strftime(
                #     "%Y-%m-%d %H:%M:%S") if with_timestamps else ""
                if isinstance(message, HumanMessage):
                    history_buffer += f"Time:{message.created_at},{user_name}: {message.content}\n"
                elif isinstance(message, AIMessage):
                    history_buffer += f"Time:{message.created_at},{role_name}: {message.content}\n"
                elif isinstance(message, SystemMessage):
                    history_buffer += f"<SYSTEM>:Time:{message.created_at}\n {message.content}\n</SYSTEM>"
            return history_buffer.strip()  # 去掉末尾换行符







    # def add_message(self, message: BaseMessage) -> None:
    #     """Append the message to the record in db"""
    #     with self.Session() as session:
    #         session.add(self.converter.to_sql_model(message, self.user_guid))
    #         session.commit()

    def add_message_with_uid(self,guid:str, message: BaseMessage) -> None:
        """Append the message to the record in db"""
        with self.Session() as session:
            session.add(self.converter.to_sql_model(message, guid))
            session.commit()

    def clear(self) -> None:
       pass


