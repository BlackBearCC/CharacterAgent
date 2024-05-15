import json
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from sqlalchemy import Column, Integer, Text, create_engine, text

from ai.models.buffer import get_prefixed_buffer_string

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict, HumanMessage, AIMessage,
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
    """
    Create a message model for a given table name.

    Args:
        table_name: The name of the table to use.
        DynamicBase: The base class to use for the model.

    Returns:
        The model class.

    """

    # Model declared inside a function to have a dynamic table name.
    class Message(DynamicBase):  # type: ignore[valid-type, misc]
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        session_id = Column(Text)
        message = Column(Text)  # Set collation to support Unicode characters

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
        if message_type == "human":
            return HumanMessage(content=content)
        elif message_type == "ai":
            return AIMessage(content=content)
        else:
            return BaseMessage(content=content)

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        return self.model_class(
            session_id=session_id, message=json.dumps(message_to_dict(message), ensure_ascii=False)  # Ensure_ascii=False to support non-ASCII characters
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


    def messages(self, count: int = 100) -> List[BaseMessage]:  # type: ignore
        """Retrieve the last 'count' messages from db, sorted by ascending id."""
        with self.Session() as session:
            result = (
                session.query(self.sql_model_class)
                .where(
                    getattr(self.sql_model_class, self.session_id_field_name)
                    == self.session_id
                )
                .order_by(self.sql_model_class.id.asc())
                .limit(count)
            )
            messages = []
            for record in result:
                messages.append(self.converter.from_sql_model(record))
            return messages

    def buffer(self, count: int = 100) -> str:
        """Retrieve the last 'count' messages from db, sorted by ascending id."""
        _messages = self.messages(10)
        history_buffer = get_prefixed_buffer_string(_messages, "大头哥", "兔几妹妹")

        return history_buffer
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


