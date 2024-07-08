import logging

from langchain_core.language_models import  BaseLanguageModel

from ai.prompts.deep_agent import DEEP_EMOTION, DEEP_INTENT
from .base_mind_module import BaseMindModule

class CognitiveModule(BaseMindModule):
    async def analyze_emotions(self, input_text):
        logging.info("Agent: Performing emotion analysis...")
        invoke_input = {"user_input": input_text}
        return await self.invoke_chain(DEEP_EMOTION, invoke_input=invoke_input, conversation_history=self.data_context_manager.messages)

    async def analyze_intent(self, input_text):
        logging.info("Agent: Performing intent recognition...")
        invoke_input = {"user_input": input_text}
        return await self.invoke_chain(DEEP_INTENT, invoke_input, conversation_history=self.data_context_manager.messages)
