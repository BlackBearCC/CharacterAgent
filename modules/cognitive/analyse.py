import asyncio
from typing import Any, Dict
from ai.prompts.deep_agent import DEEP_EMOTION, DEEP_INTENT, DEEP_CONTEXT

class AnalyseModule:
    def __init__(self, invoke_chain_func):
        self.invoke_chain = invoke_chain_func

    async def analyze(self, focused_data: Dict[str, Any],input_text:str, memory_context: str) -> Dict[str, Any]:
        input_text = focused_data.get("processed_input", {}).get("user_input", "")
        
        tasks = [
            self._analyze_emotion(input_text,memory_context),
            self._analyze_intent(input_text,memory_context),
            self._analyze_context(input_text,memory_context)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        emotions, intent, context = [result for result in results if result is not None]

        return {
            "emotion": emotions if isinstance(emotions, str) else "Error",
            "intent": intent if isinstance(intent, str) else "Error",
            "context": context if isinstance(context, str) else "Error"
        }

    async def _analyze_emotion(self, input_text: str,memory_context:str) -> str:
        return await self.invoke_chain("analyze_emotions", DEEP_EMOTION, input_text,memory_context)

    async def _analyze_intent(self, input_text: str,memory_context:str) -> str:
        return await self.invoke_chain("analyze_intent", DEEP_INTENT, input_text,memory_context)

    async def _analyze_context(self, input_text: str,memory_context:str) -> str:
        return await self.invoke_chain("extract_key_content", DEEP_CONTEXT, input_text,memory_context)