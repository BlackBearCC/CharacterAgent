import asyncio

from langchain_core.language_models import BaseLanguageModel

from modules.cognitive_module import CognitiveModule
from modules.data_context import DataContextManager
from modules.thinking_moudul import ThinkingModule


class AIAgent:
    def __init__(self, data_context: DataContextManager):
        self.cognitive_module = CognitiveModule(data_context)
        self.thinking_module = ThinkingModule(data_context)
        # self.planning_module = PlanningModule(data_context)
        # self.thinking_module = ThinkingModule(data_context)
        # self.behavior_module = BehaviorModule(data_context)
        # self.constraint_module = ConstraintModule(data_context)
        # self.reflection_module = ReflectionModule(data_context)

    async def process_input (self, input_text: str,user_id:str) -> str:
        ##认知模块任务##
        await self.cognitive_module.process_output(input_text, user_id)
        emotion = self.cognitive_module.get_emotion_analysis()
        intent = self.cognitive_module.get_intent_analysis()
        context = self.cognitive_module.get_context_analysis()
        character_profile = self.cognitive_module.get_character_profile()

        # ##思考模块任务##
        # thinking_output = await self.thinking_module.process_output(
        #     emotion, intent, context, character_profile
        # )

        ##行为模块任务##
        return  character_profile