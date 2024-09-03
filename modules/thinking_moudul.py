from typing import Any, Dict, List, Union
from modules.base_mind_module import BaseMindModule
from modules.data_context import DataContextManager
import random

class ThinkingModule(BaseMindModule):
    def __init__(self, data_context_manager: DataContextManager):
        super().__init__(data_context_manager)
        
        self.thinking_factors = {
            "逻辑因子": self._logical_thinking,
            "情感因子": self._emotional_thinking,
            "创造因子": self._creative_thinking,
            "分析因子": self._analytical_thinking
        }

    async def process_output(self, emotion: str, intent: str, context: str, character_profile: Dict[str, Any]) -> Dict[str, Any]:
        strategy = self._choose_strategy()
        
        if strategy == "简单策略":
            thought_process = await self._simple_thinking(emotion, intent, context, character_profile)
            response_strategy = self._determine_response_strategy(thought_process)
        else:  # 深度策略
            selected_factors = self._select_thinking_factors()
            thought_processes = await self._deep_thinking(selected_factors, emotion, intent, context, character_profile)
            response_strategy = self._determine_response_strategy(thought_processes)

        return {
            "strategy": strategy,
            "thought_processes": thought_processes if strategy == "深度策略" else thought_process,
            "response_strategy": response_strategy,
            "selected_factors": selected_factors if strategy == "深度策略" else None
        }


    def _choose_strategy(self) -> str:
        return random.choice(["简单策略", "深度策略"])

    def _select_thinking_factors(self) -> List[str]:
        return random.sample(list(self.thinking_factors.keys()), k=random.randint(1, len(self.thinking_factors)))

    async def _simple_thinking(self, emotion: str, intent: str, context: str, character_profile: Dict[str, Any]) -> str:
        prompt = f"""
        基于以下信息进行简单思考：
        用户情绪: {emotion}
        用户意图: {intent}
        对话上下文: {context}
        
        考虑我的角色特征：{character_profile['personality']}
        
        请生成一个简短的思考过程，分析用户的需求和可能的回应方式。
        """
        return await self.invoke_chain("simple_thinking", prompt, {"emotion": emotion, "intent": intent, "context": context})

    async def _deep_thinking(self, selected_factors: List[str], emotion: str, intent: str, context: str, character_profile: Dict[str, Any]) -> Dict[str, str]:
        thought_processes = {}
        for factor in selected_factors:
            thought_processes[factor] = await self.thinking_factors[factor](emotion, intent, context)
        return thought_processes

    async def _logical_thinking(self, emotion: str, intent: str, context: str) -> str:
        prompt = f"""
        从逻辑角度分析以下情况：
        用户情绪: {emotion}
        用户意图: {intent}
        对话上下文: {context}
        
        请给出一个逻辑分析和推理过程。
        """
        return await self.invoke_chain("logical_thinking", prompt, {"emotion": emotion, "intent": intent, "context": context})

    async def _emotional_thinking(self, emotion: str, intent: str, context: str) -> str:
        prompt = f"""
        从情感角度分析以下情况：
        用户情绪: {emotion}
        用户意图: {intent}
        对话上下文: {context}
        
        请给出一个情感分析和共情过程。
        """
        return await self.invoke_chain("emotional_thinking", prompt, {"emotion": emotion, "intent": intent, "context": context})

    async def _creative_thinking(self, emotion: str, intent: str, context: str) -> str:
        prompt = f"""
        从创造性角度思考以下情况：
        用户情绪: {emotion}
        用户意图: {intent}
        对话上下文: {context}
        
        请给出一些创新的想法或解决方案。
        """
        return await self.invoke_chain("creative_thinking", prompt, {"emotion": emotion, "intent": intent, "context": context})

    async def _analytical_thinking(self, emotion: str, intent: str, context: str) -> str:
        prompt = f"""
        从分析角度思考以下情况：
        用户情绪: {emotion}
        用户意图: {intent}
        对话上下文: {context}
        
        请给出一个详细的情况分析和可能的行动建议。
        """
        return await self.invoke_chain("analytical_thinking", prompt, {"emotion": emotion, "intent": intent, "context": context})

    def _determine_response_strategy(self, thought_processes: Union[str, Dict[str, str]]) -> Dict[str, Any]:
        if isinstance(thought_processes, str):
            # 简单策略的情况
            if "情绪低落" in thought_processes:
                return {"tone": "安慰", "focus": "情感支持"}
            elif "寻求信息" in thought_processes:
                return {"tone": "专业", "focus": "提供信息"}
            elif "需要建议" in thought_processes:
                return {"tone": "建议性", "focus": "给出建议"}
            else:
                return {"tone": "中性", "focus": "一般交流"}
        else:
            # 深度策略的情况
            combined_thoughts = " ".join(thought_processes.values())
            if "情感" in combined_thoughts:
                return {"tone": "共情", "focus": "情感支持"}
            elif "逻辑" in combined_thoughts:
                return {"tone": "理性", "focus": "逻辑分析"}
            elif "创新" in combined_thoughts:
                return {"tone": "启发性", "focus": "创新思路"}
            elif "分析" in combined_thoughts:
                return {"tone": "深入", "focus": "详细分析"}
            else:
                return {"tone": "平衡", "focus": "综合考虑"}

