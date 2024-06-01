import logging
from datetime import datetime
from typing import List

from langchain_core.messages import BaseMessage
from sqlalchemy.orm import scoped_session

from ai.models.ai import AIMessage
from ai.models.human import HumanMessage
from ai.models.system import SystemMessage
from .models import Base, Message


class MessageMemory:
    def __init__(self, session: scoped_session):
        self.session = session


    def add_message(self, message: Message):
        """Append the message to the record in db"""
        try:
            self.session.add(message)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logging.error(f"Failed to add message: {e}")
            raise

    def get_messages(self, guid: str, count: int = 100) -> List[Message]:
        """Retrieve messages from the database."""
        try:
            return (self.session.query(Message)
                    .filter(Message.user_guid == guid)
                    .order_by(Message.created_at.desc())
                    .limit(count)
                    .all())
        except Exception as e:
            logging.error(f"Failed to retrieve messages: {e}")
            raise

    def buffer(self, guid: str, user_name="主人", role_name="兔兔", count: int = 100,
               with_timestamps: bool = True) -> str:
        messages = self.get_messages(guid, count)
        buffer = ""
        for message in messages:
            buffer += self.format_message(message, user_name, role_name)
        return buffer.strip()

    def buffer_with_time(self, guid: str, date_start: datetime, date_end: datetime, user_name="主人", role_name="兔兔",
                         count: int = 300) -> str:
        messages = self.get_messages(guid, count)
        buffer = ""
        for message in messages:
            if date_start <= message.created_at <= date_end:
                buffer += self.format_message(message, user_name, role_name)
        return buffer.strip()

    def format_message(self, message, user_name, role_name):
        if isinstance(message, HumanMessage):
            return f"Time:{message.created_at},{user_name}: {message.content}\n"
        elif isinstance(message, AIMessage):
            return f"Time:{message.created_at},{role_name}: {message.content}\n"
        elif isinstance(message, SystemMessage):
            return f"<SYSTEM>:Time:{message.created_at}\n {message.content}\n</SYSTEM>"



