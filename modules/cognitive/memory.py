import time
import redis
from typing import Any, Dict, List
from mem0 import MemoryClient
import json
import uuid

class MemoryModule:
    def __init__(self, api_key: str):
        self.client = MemoryClient(api_key=api_key)
        self.redis = redis.Redis(host='182.254.242.30', port=6379, db=0,password="669988")
        try:
            self.redis = redis.Redis(host='182.254.242.30', port=6379, db=0, password='669988')
            self.redis.ping()
            print("成功连接到Redis")
        except redis.ConnectionError as e:
            print(f"无法连接到Redis: {e}")
            self.redis = None        

    def get_or_create_session(self, user_id: str) -> str:
        current_time = int(time.time())
        last_activity_key = f"last_activity:{user_id}"
        last_activity = self.redis.get(last_activity_key)

        if last_activity is None or (current_time - int(last_activity)) > 4 * 3600:
            session_id = f"{user_id}:{current_time}"
            self.redis.set(f"current_session:{user_id}", session_id)
        else:
            session_id = self.redis.get(f"current_session:{user_id}").decode()

        self.redis.set(last_activity_key, current_time)
        return session_id
        

    def retrieve(self, query: str, user_id: str = None, agent_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """
        搜索记忆,不支持异步方法
        """
        print(f"查询记忆：{query},id:{user_id}")
        filters = {"user_id":user_id}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if session_id:
            filters["session_id"] = session_id

        if not filters:
            raise ValueError("无有效用户过滤条件")
        return  self.client.search(query,  user_id=user_id)
    

    def update(self, messages: List[Dict[str, str]], user_id: str = None, agent_id: str = None, session_id: str = None):
        self.client.add(messages, user_id=user_id, agent_id=agent_id, session_id=session_id)
      

    def get_all_memories(self, user_id: str = None, agent_id: str = None, session_id: str = None) -> List[Dict[str, Any]]:
        return  self.client.get_all(user_id=user_id, agent_id=agent_id, session_id=session_id)

    async def delete_memories(self, user_id: str = None, agent_id: str = None, session_id: str = None):
        if user_id:
            await self.client.delete_all(user_id=user_id)
        elif agent_id:
            await self.client.delete_all(agent_id=agent_id)
        elif session_id:
            await self.client.delete_all(session_id=session_id)

    def get_memory_history(self, memory_id: str) -> List[Dict[str, Any]]:
        return  self.client.history(memory_id)

    async def update_specific_memory(self, memory_id: str, new_content: str):
        await self.client.update(memory_id, new_content)

    async def delete_specific_memory(self, memory_id: str):
        await self.client.delete(memory_id)

    async def get_all_users(self) -> List[str]:
        return await self.client.users()
    

    def add_normal_memorise(self, user_id: str, role: str, content: str):
        session_id = self.get_or_create_session(user_id)
        timestamp = int(time.time() * 1000)
        message_id = f"{user_id}:{session_id}:{timestamp}"
        
        message_data = {
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        self.redis.hset(f"message:{message_id}", mapping=message_data)
        self.redis.zadd(f"chat:{user_id}:{session_id}", {message_id: timestamp})

    def get_normal_memorise(self, user_id: str, start: int = 0, end: int = -1) -> List[Dict[str, Any]]:
        session_id = self.get_or_create_session(user_id)
        message_ids = self.redis.zrange(f"chat:{user_id}:{session_id}", start, end)
        
        messages = []
        for message_id in message_ids:
            message_data = self.redis.hgetall(f"message:{message_id.decode()}")
            messages.append({k.decode(): v.decode() for k, v in message_data.items()})
        
        return messages


    def end_session(self, user_id: str, session_id: str):
        # 将会话数据保存到长期存储的逻辑
        pass