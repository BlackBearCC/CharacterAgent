import logging
from datetime import datetime
from typing import List

from langchain_core.messages import BaseMessage
from sqlalchemy.orm import scoped_session

from ai.models.ai import AIMessage
from ai.models.human import HumanMessage
from ai.models.system import SystemMessage
from data.database.mysql.models import Message


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

    def add_messages(self, messages):
        """Append the message to the record in db"""
        try:
            self.session.bulk_save_objects(messages)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logging.error(f"Failed to batch add message: {e}")
            raise


    def get_messages(self, guid, count=100, start_date=None, end_date=None):
        query = self.session.query(Message).filter(Message.user_guid == guid)
        if start_date and end_date:
            query = query.filter(Message.created_at >= start_date, Message.created_at <= end_date)
        return query.order_by(Message.created_at.desc()).limit(count).all()

    def buffer_messages(self, guid, user_name="主人", role_name="兔兔", count=100, start_date=None, end_date=None):
        messages = self.get_messages(guid, count, start_date, end_date)
        return '\n'.join(self.format_message(m, user_name, role_name) for m in messages)

    def format_message(self, message, user_name, role_name):
        # 确保此处逻辑符合实际需要
        if message.type == "human":
            return f"EventTime:{message.created_at}, {user_name}: {message.message}"
        elif message.type == "ai":
            return f"EventTime:{message.created_at}, {role_name}: {message.message}"
        elif message.type == "system":
            return f"<SYSTEM> EventTime:{message.created_at}\n{message.message}</SYSTEM>"
        else:
            return f"NORMAL: {message.message}"

