import logging
import re
from datetime import datetime
from typing import List, Tuple

from langchain_core.messages import BaseMessage
from sqlalchemy import func
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
        # 反转消息列表，使其从最早的消息开始
        reversed_messages = reversed(messages)
        return '\n'.join(self.format_message(m, user_name, role_name) for m in reversed_messages)

    def format_message(self, message, user_name, role_name):
        # 这是一个示例格式化函数，具体实现可能根据你的需求有所不同
        return f"{message.created_at.strftime('%Y-%m-%d %H:%M:%S')} - [{role_name} ({user_name})]: {message.message}"

    async def check_and_buffer_messages(self, guid: str, user_name: str, role_name: str, threshold: int = 10,
                                        start_date=None, end_date=None) -> Tuple[str, List[int]]:
        """
        检查用户的消息数量，如果达到或超过指定阈值，则返回缓冲的消息内容及消息ID列表。
        :param guid: 用户唯一标识
        :param user_name: 用户名
        :param role_name: 角色名
        :param threshold: 消息数量阈值
        :param start_date: 查询的起始日期‘’
        :param end_date: 查询的结束日期
        :return: 达到阈值时的消息缓冲内容及消息ID列表，否则返回空字符串和空列表
        """
        query = self.session.query(Message.id, Message.message).filter(
            Message.user_guid == guid,
            Message.summary_id.is_(None),  # 确保只选取还没有摘要的消息
            Message.created_at >= start_date if start_date else True,
            Message.created_at <= end_date if end_date else True
        ).order_by(Message.created_at.desc()).limit(threshold)  # 注意这里改为倒序，以保证最新的消息被优先处理

        messages = []
        message_ids = []
        for msg_id, content in query:
            messages.append(content)
            message_ids.append(msg_id)

        if len(messages) >= threshold:
            return "\n".join(messages), message_ids  # 返回消息内容和ID列表
        else:
            return "", []
    def bind_summary_id_to_messages(self, message_ids: List[int], summary_id: int) -> None:
        for msg_id in message_ids:
            message = self.session.query(Message).filter(Message.id == msg_id).first()

            if message:
                print(f"Before setting summary_id: {message.summary_id}")
                message.summary_id = summary_id
                print(f"After setting summary_id: {message.summary_id}")
            else:
                logging.error(f"Message with ID {msg_id} not found.")
        self.session.commit()
        logging.info(f"Summary ID {summary_id} bound to messages {message_ids}.")

    def format_message(self, message, user_name, role_name):
        # 使用正则表达式移除消息中的大括号{}
        cleaned_message = re.sub(r'\{|\}', '', message.message)
        if message.type == "human":
            return f"EventTime:{message.created_at}, {user_name}: {cleaned_message}"
        elif message.type == "ai":
            return f"EventTime:{message.created_at}, {role_name}: {cleaned_message}"
        elif message.type == "system":
            return f"<SYSTEM> EventTime:{message.created_at}\n{cleaned_message}</SYSTEM>"
        else:
            return f"NORMAL: {cleaned_message}"
