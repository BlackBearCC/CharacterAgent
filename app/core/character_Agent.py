from typing import Any, Dict, List

from app.core import AbstractAgent


class CharacterAgent(AbstractAgent):
    def response(self, input_text: str) -> str:
        try:
            response = self.model.generate_response(input_text, self.memory)
            return response
        except Exception as e:
            # 简单处理异常，实际应用中可能需要更复杂的错误处理逻辑
            print(f"生成响应时发生错误: {e}")
            raise

    def perform_task(self, task: str, data: dict) -> int:
        return 200

    def remember(self, key: str):
        return 200

    def recall(self, key: str) -> any:
        pass

    def __init__(self, name: str, model: Any):
        self.name = name
        self.model = model
        self.memory = {}


