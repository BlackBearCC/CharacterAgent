
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, text
from datetime import datetime
import pytz

beijing_tz = pytz.timezone('Asia/Shanghai')
now_beijing_time = datetime.now(beijing_tz)


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
    created_at = Column(DateTime, default=datetime.now(beijing_tz))
    updated_at = Column(DateTime, default=datetime.now(beijing_tz), onupdate=datetime.now(beijing_tz))
    user_guid = Column(String(128), ForeignKey('user.guid'))


class Message(Base):
    """Data model for message."""
    __tablename__ = "message_store"
    id = Column(Integer, primary_key=True)
    user_guid = Column(String(128), ForeignKey('user.guid'))
    type = Column(String)
    role = Column(String)
    message = Column(Text)
    created_at = Column(DateTime, nullable=False, default=datetime.now(beijing_tz),server_default=text('CURRENT_TIMESTAMP'))  # Corrected
    generate_from = Column(Text)
    call_step = Column(Text)

