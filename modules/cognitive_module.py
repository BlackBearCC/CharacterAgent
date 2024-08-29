import os
from typing import Any, Dict

from dotenv import load_dotenv
from modules.base_mind_module import BaseMindModule
from modules.data_context import DataContextManager
from modules.cognitive import PerceptionModule, MemoryModule, CharacterModule, AttentionModule, AnalyseModule

class CognitiveModule(BaseMindModule):
    def __init__(self, data_context_manager: DataContextManager):
        super().__init__(data_context_manager)
        load_dotenv()
        # openai_key = os.getenv("OPENAI_API_KEY")
        memo_api_key = os.getenv('MEMOAI_API_KEY')
        self.perception = PerceptionModule()
        self.memory = MemoryModule(api_key=memo_api_key)
        self.character = CharacterModule()
        self.attention = AttentionModule()
        self.analyse = AnalyseModule(self.invoke_cognitive_chain)

    async def process_input(self, input_text: str,user_id:str) -> Dict[str, Any]:
        print(f"处理输入：{input_text}")
        # 添加用户输入到短期记忆
        self.memory.add_normal_memorise(user_id, "大头", input_text)
        perceived_data = await self.perception.process_input(input_text)
        # focused_data = self.attention.focus(perceived_data)
        memory_context =  self.memory.get_normal_memorise(user_id=user_id)
        print(f"常规记忆历史：{memory_context}")
        
        # analysis_results = await self.analyse.analyze(focused_data, memory_context)
        
        
        # character_profile = self.character.get_character_profile()
        # self.memory.update(analysis_results,user_id=123)
        return {
            "analysis": "analysis_results",
            # "character_profile": character_profile,
            # "memory_context": memory_context
        }
    async def invoke_cognitive_chain(self, chain_type: str, prompt_template, input_text: str) -> Any:
        invoke_input = {"user_input": input_text}
        try:
            return await self.invoke_chain(prompt_template, invoke_input=invoke_input, conversation_history=self.data_context_manager.messages)
        except Exception as e:
            return f"Error processing {chain_type}: {str(e)}"