from langchain_core.language_models import BaseLanguageModel

from modules.cognitive_module import CognitiveModule
from modules.data_context import DataContextManager


class AIAgent:
    def __init__(self, data_context: DataContextManager):
        self.cognitive_module = CognitiveModule(data_context)
        # self.planning_module = PlanningModule(data_context)
        # self.thinking_module = ThinkingModule(data_context)
        # self.behavior_module = BehaviorModule(data_context)
        # self.constraint_module = ConstraintModule(data_context)
        # self.reflection_module = ReflectionModule(data_context)

    async def process_input(self, input_text):
        emotions = await self.cognitive_module.analyze_emotions( input_text)
        intent = await self.cognitive_module.analyze_intent( input_text)
        # Add further module processing here
        response = f"Emotion: {emotions}\nIntent: {intent}"
        return response