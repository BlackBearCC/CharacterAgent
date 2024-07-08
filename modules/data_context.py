# 数据上下文管理器
from app.service.services import DBContext


class DataContextManager:
    def __init__(self, db_context: DBContext, uid: str):
        self.db_context = db_context
        self.uid = uid
        print("DataContextManager init")
        self.messages = self.fetch_messages()

    def fetch_messages(self, count=10):
        return self.db_context.message_memory.buffer_messages(guid=self.uid, count=count)