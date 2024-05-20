import datetime
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from sqlalchemy import Column, Integer, Text, create_engine, text

from ai.models.ai import AIMessage
from ai.models.buffer import get_prefixed_buffer_string
from ai.models.human import HumanMessage

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
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
    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
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
        session_id = Column(Text)
        message = Column(Text)  # Set collation to support Unicode characters
        created_at = Column(Integer)  # Add a timestamp column (assuming Unix timestamp in seconds)

    return Message


class DefaultMessageConverter(BaseMessageConverter):
    """The default message converter for SQLChatMessageHistory."""

    def __init__(self, table_name: str):
        self.model_class = create_message_model(table_name, declarative_base())
        # with self.engine.connect() as connection:
        #     connection.execute(
        #         text(f"ALTER TABLE {table_name} CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"))

    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        message_dict = json.loads(sql_message.message)
        message_type = message_dict.get("type")
        content = message_dict.get("data", {}).get("content", "")
        created_at = sql_message.created_at  # Get the timestamp from the SQL model

        if message_type == "human":
            message = HumanMessage(content=content, created_at=created_at)
        elif message_type == "ai":
            message = AIMessage(content=content, created_at=created_at)
        else:
            message = BaseMessage(content=content, created_at=created_at)

        # Add the created_at attribute to the message dict
        message_dict["data"]["created_at"] = created_at
        message.message = json.dumps(message_dict, ensure_ascii=False)

        return message

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        timestamp = int(time.time())  # Get current Unix timestamp
        return self.model_class(
            session_id=session_id,
            message=json.dumps(message_to_dict(message), ensure_ascii=False),
            created_at=timestamp,  # Add created_at field
        )
    def get_sql_model_class(self) -> Any:
        return self.model_class


class SQLChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in an SQL database."""

    def __init__(
        self,
        session_id: str,
        connection_string: str,
        table_name: str = "message_store",
        session_id_field_name: str = "session_id",
        custom_message_converter: Optional[BaseMessageConverter] = None,
        _create_table_if_not_exists: bool = True,

    ):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, echo=False)
        self.session_id_field_name = session_id_field_name
        self.converter = DefaultMessageConverter(table_name)
        self.sql_model_class = self.converter.get_sql_model_class()
        if not hasattr(self.sql_model_class, session_id_field_name):
            raise ValueError("SQL model class must have session_id column")
        self._create_table_if_not_exists()

        self.session_id = session_id
        self.Session = sessionmaker(self.engine)


    def _create_table_if_not_exists(self) -> None:
        self.sql_model_class.metadata.create_all(self.engine)


    def messages(self, count: int = 100) -> List[BaseMessage]:
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
                .where(getattr(self.sql_model_class, self.session_id_field_name) == self.session_id)
                .order_by(self.sql_model_class.id.desc())
                .limit(count)
                .all()  # 添加.all()来执行查询
            )


            messages = []  # 初始化消息列表

            # 反向遍历查询结果，并将每条记录转换为BaseMessage对象，添加到消息列表中
            for record in reversed (result):
                messages.append(self.converter.from_sql_model(record))

            return messages  # 返回消息列表

    def buffer(self, count: int = 100, with_timestamps: bool = True) -> str:
        try:
            _messages = self.messages(count)
            history_buffer = ""

            for message in _messages:
                timestamp = datetime.datetime.fromtimestamp(message.created_at).strftime(
                    "%Y-%m-%d %H:%M:%S") if with_timestamps else ""
                if isinstance(message, HumanMessage):
                    history_buffer += f"{timestamp} 大头哥: {message.content}\n"
                elif isinstance(message, AIMessage):
                    history_buffer += f"{timestamp} 兔几妹妹: {message.content}\n"
            print(history_buffer.strip())
            return history_buffer.strip()  # 去掉末尾换行符
        except Exception as e:
            logger.error(f"Error occurred while fetching messages: {str(e)}")  # 记录错误日志
            return "No messages found"  # 返回无消息提示






    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in db"""
        with self.Session() as session:
            session.add(self.converter.to_sql_model(message, self.session_id))
            session.commit()

    def clear(self) -> None:
        """Clear session memory from db"""

        with self.Session() as session:
            session.query(self.sql_model_class).filter(
                getattr(self.sql_model_class, self.session_id_field_name)
                == self.session_id
            ).delete()
            session.commit()


