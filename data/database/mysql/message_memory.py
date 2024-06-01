from sqlalchemy.orm import scoped_session
from .models import Base, Message


class MessageMemory:
    def __init__(self, session: scoped_session):
        self.session = session

    def add_message(self, message: Message):
        """Add a new message to the database."""
        self.session.add(message)
        self.session.commit()

    def get_messages(self, user_guid: str, count: int = 100):
        """Retrieve the latest messages for a given user GUID."""
        return (self.session.query(Message)
                .filter_by(user_guid=user_guid)
                .order_by(Message.id.desc())
                .limit(count)
                .all())

    def get_message_by_user_guid(self, user_guid: str):
        """Retrieve messages by user GUID."""
        return (self.session.query(Message)
                .filter_by(user_guid=user_guid)
                .order_by(Message.id.desc())
                .all())

    def delete_messages_by_user_guid(self, user_guid: str):
        """Delete messages by user GUID."""
        messages = self.session.query(Message).filter_by(user_guid=user_guid).all()
        for message in messages:
            self.session.delete(message)
        self.session.commit()




