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
from data.database.mysql.models import Message, Message_Summary


class MessageSummary:
    def __init__(self, session: scoped_session):
        self.session = session

    def add_summary(self, message_summary: Message_Summary, message_ids: List[int]=None) -> int:
        try:
            if message_ids:
                self.session.add(message_summary)
                self.session.flush()  # 使用 flush 来确保 message_summary 获取到 ID，但不提交整个事务
                summary_id = message_summary.id

                messages = self.session.query(Message).filter(Message.id.in_(message_ids)).all()
                for message in messages:
                    message.summary_id = summary_id

                self.session.commit()
                return summary_id
            else:
                self.session.add(message_summary)
                self.session.commit()
                return message_summary.id

        except Exception as e:
            self.session.rollback()
            raise Exception(f"Failed to add summary and bind messages: {e}")


    def get_summaries_within_period(self, session, guid, start_date=None, end_date=None, max_count=100):
        """
        Retrieve a list of summaries for a specific user within a given date range,
        ordered from the most recent to the oldest, limited by max_count.
        :param session: The SQLAlchemy session to use for the query
        :param guid: User GUID to filter the summaries
        :param start_date: Optional start date to filter summaries
        :param end_date: Optional end date to filter summaries
        :param max_count: Maximum number of summaries to return
        :return: A list of Message_Summary instances
        """
        query = session.query(Message_Summary).filter(Message_Summary.user_guid == guid)

        if start_date:
            query = query.filter(Message_Summary.created_at >= start_date)
        if end_date:
            query = query.filter(Message_Summary.created_at <= end_date)

        return query.order_by(Message_Summary.created_at.desc()).limit(max_count).all()

    def buffer_summaries(self, guid, max_count=100,start_date=None, end_date=None, ):
        """
        Retrieves and concatenates summaries into a single string, including their creation dates.
        :param session: The SQLAlchemy session to use for the query
        :param guid: User GUID
        :param start_date: Optional start date to filter summaries
        :param end_date: Optional end date to filter summaries
        :param max_count: Maximum number of summaries to return
        :return: A string concatenating all summaries with their creation dates
        """
        summaries = self.get_summaries_within_period(self.session, guid, start_date, end_date, max_count)
        reversed_summaries = reversed(summaries)
        return "\n".join(
            [f"{summary.created_at.strftime('%Y-%m-%d %H:%M:%S')} - {summary.summary}" for summary in reversed_summaries])