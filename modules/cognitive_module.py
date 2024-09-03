import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from modules.base_mind_module import BaseMindModule
from modules.data_context import DataContextManager
from modules.cognitive.character import CharacterModule
from modules.cognitive.perception import PerceptionModule
from modules.cognitive.analyse import AnalyseModule
from modules.cognitive.attention import AttentionModule
from modules.cognitive.memory import MemoryModule

class CognitiveModule(BaseMindModule):
    def __init__(self, data_context_manager: DataContextManager):
        super().__init__(data_context_manager)
        load_dotenv()
        # openai_key = os.getenv("OPENAI_API_KEY")
        memo_api_key = os.getenv('MEMOAI_API_KEY')
        self.data_context = data_context_manager
        self.character = CharacterModule()
        self.perception = PerceptionModule()
        self.analyse = AnalyseModule(self.invoke_cognitive_chain)
        self.attention = AttentionModule()
        self.memory = MemoryModule(api_key=memo_api_key)
        
        self.cognitive_state: Dict[str, Any] = {}
        self.last_input: Optional[str] = None
        self.last_user_id: Optional[str] = None

    async def process_output(self, input_text: str, user_id: str) -> Dict[str, Any]:
        self.last_input = input_text
        self.last_user_id = user_id
        
        self.memory.add_normal_memorise(user_id, "用户", input_text)
        perceived_data = await self.perception.process_input(input_text)
        focused_data = self.attention.focus(perceived_data)
        memory_context = self.memory.get_normal_memorise(user_id=user_id)
        
        analysis_results = await self.analyse.analyze(focused_data, input_text, memory_context)
        
        self.cognitive_state = {
            "character_profile": self.character.get_character_profile(),
            "perceived_data": perceived_data,
            "focused_data": focused_data,
            "analysis_results": analysis_results,
            "memory_context": memory_context
        }
        
        return self.cognitive_state

    def get_cognitive_state(self) -> Dict[str, Any]:
        return self.cognitive_state

    def get_character_profile(self) -> Dict[str, Any]:
        try:
            return self.character.get_character_profile()
        except Exception as e:
            print(f"获取角色档案时出错: {e}")
            return "无"

    def get_emotion_analysis(self) -> Optional[str]:
        try:
            return self.cognitive_state.get("analysis_results", {}).get("emotion")
        except Exception as e:
            print(f"获取情感分析结果时出错: {e}")
            return "无"

    def get_intent_analysis(self) -> Optional[str]:
        try:
            return self.cognitive_state.get("analysis_results", {}).get("intent")
        except Exception as e:
            print(f"获取意图分析结果时出错: {e}")
            return "无"

    def get_context_analysis(self) -> Optional[str]:
        try:
            return self.cognitive_state.get("analysis_results", {}).get("context")
        except Exception as e:
            print(f"获取上下文分析结果时出错: {e}")
            return "无"

    def get_memory_context(self) -> str:
        try:
            return self.cognitive_state.get("memory_context", "")
        except Exception as e:
            print(f"获取记忆上下文时出错: {e}")
            return "无"

    async def refresh_analysis(self) -> None:
        if self.last_input and self.last_user_id:
            await self.process_input(self.last_input, self.last_user_id)
        else:
            print("无法重新分析，因为没有上一次的输入信息")

    async def invoke_cognitive_chain(self, chain_type: str, prompt_template, memory_context: str, input_text: str) -> Any:
        ## 模块自有链，调用父类invoke_chain生成特殊内容
        invoke_input = {"user_input": input_text}
        try:
            return await self.invoke_chain(prompt_template, invoke_input=invoke_input, conversation_history=memory_context)
        except Exception as e:
            print(f"处理 {chain_type} 时出错: {str(e)}")
            return None