
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, text
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    __tablename__ = 'user'
    guid = Column(String(128), primary_key=True)
    username = Column(String(50), server_default='主人')  # 默认用户名
    role_name = Column(String(50), server_default='兔兔')  # 默认角色名
    email = Column(String(100))
    game_uid = Column(String(16))

class Entity(Base):
    """Data model for entity."""
    __tablename__ = 'entity_store'
    entity_id = Column(Integer, primary_key=True)
    entity = Column(String(255))
    summary = Column(String(255))
    created_at = Column(DateTime, server_default="CURRENT_TIMESTAMP")
    updated_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))


    user_guid = Column(String(128), ForeignKey('user.guid'))


class Message(Base):
    """Data model for message."""
    __tablename__ = "message_store"
    id = Column(Integer, primary_key=True)
    user_guid = Column(String(128), ForeignKey('user.guid'))
    type = Column(String)
    role = Column(String)
    message = Column(Text)
    created_at = Column(DateTime, nullable=False, server_default="CURRENT_TIMESTAMP")
    generate_from = Column(Text)
    call_step = Column(Text)
    # response_metadata = Column(Text)
    summary_id = Column(Integer, ForeignKey('message_summary_store.id'))  # 添加这一行



class Message_Summary(Base):
    """Data model for message."""
    __tablename__ = "message_summary_store"
    id = Column(Integer, primary_key=True,autoincrement=True)
    user_guid = Column(String(128), ForeignKey('user.guid'))
    summary = Column(String)
    created_at = Column(DateTime, server_default="CURRENT_TIMESTAMP")
    # messages = relationship("Message", back_populates="summary")  # 添加反向关系