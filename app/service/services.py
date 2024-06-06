from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from dotenv import load_dotenv
import os

from data.database.mysql.entity_memory import EntityMemory
from data.database.mysql.message_memory import MessageMemory
from data.database.mysql.message_summary import MessageSummary
from data.database.mysql.models import Message_Summary
from data.database.mysql.user_management import UserDatabase
load_dotenv()
# 创建全局数据库引擎
DATABASE_CONNECTION_STRING = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_CONNECTION_STRING,pool_size=16, max_overflow=100, echo=False)

# 创建全局会话工厂
SessionLocal = scoped_session(sessionmaker(bind=engine))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_database():
    db = next(get_db())
    return UserDatabase(db)

def get_message_memory():
    db = next(get_db())
    return MessageMemory(db)

def get_entity_memory():
    db = next(get_db())
    return EntityMemory(db)

def get_message_summary():
    db = next(get_db())
    return MessageSummary(db)

@dataclass
class DBContext:
    user_db: UserDatabase = None
    message_memory: MessageMemory = None
    entity_memory: EntityMemory = None
    message_summary: MessageSummary = None