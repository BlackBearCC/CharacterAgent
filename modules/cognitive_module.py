import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from mem0 import Memory, MemoryClient
from transformers import pipeline

from ai.prompts.deep_agent import DEEP_EMOTION, DEEP_INTENT, DEEP_CONTEXT
from modules.base_mind_module import BaseMindModule
from modules.data_context import DataContextManager


class CognitiveModule(BaseMindModule):
    def __init__(self, data_context_manager: DataContextManager):
        super().__init__(data_context_manager)
        # self.classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    async def invoke_cognitive_chain(self, chain_type: str,prompt_templet, input_text: str) -> Any:
        """
        调用指定的chain类型，并返回结果。
        """
        logging.info(f"Agent: Performing {chain_type}...")
        invoke_input = {"user_input": input_text}
        try:
            return await self.invoke_chain(prompt_templet, invoke_input=invoke_input, conversation_history=self.data_context_manager.messages)
        except Exception as e:
            logging.error(f"Error processing {chain_type}: {e}")
            return None

    async def analyze_emotions(self, input_text: str) -> Any:
        return await self.invoke_cognitive_chain("analyze_emotions",DEEP_EMOTION, input_text)

    def machine_analyze_emotion(self,input_text:str):
        """
        使用transformers库的pipeline函数创建一个分类器，并使用它对输入文本进行情感分析。
        """
        from transformers import pipeline


        sequence_to_classify = input_text
        candidate_labels = ["状态", "情感", "工作", "喜好"]
        output = self.classifier(sequence_to_classify, candidate_labels, multi_label=False)
        print(output)
        # m = Memory()
        # messages = [
        #     {"role": "user", "content": "我喜欢兔子"},
        #     {"role": "assistant",
        #      "content": "好的，兔子很可爱"}
        # ]

        # result = client.add(  messages, user_id="datou"
        #
        #                     )
        # print(result)
        # result = client.search("我喜欢什么",user_id="datou")
        # print(f"搜索结果：{result}")

        return output

    async def memo_ai(self, input_text: str) -> Any:
        # messages = [
        #     {"role": "user", "content": "我喜欢兔子"},
        #     {"role": "assistant",
        #      "content": "好的，兔子很可爱"}
        # ]
        load_dotenv()
        # openai_key = os.getenv("OPENAI_API_KEY")

        memo_api_key = os.getenv('MEMOAI_API_KEY')
        logging.info(f"Agent: Performing memo_ai...")
        client = MemoryClient(api_key=memo_api_key)
        # result = client.add(messages, user_id="datou")
        result = client.search("我喜欢什么", user_id="datou")
        print(f"搜索结果：{result}")
        return result
    async def analyze_intent(self, input_text: str) -> Any:
        return await self.invoke_cognitive_chain("analyze_intent",DEEP_INTENT, input_text)

    async def extract_key_content(self, input_text: str) -> Any:
        return await self.invoke_cognitive_chain("extract_key_content",DEEP_CONTEXT, input_text)

    async def process_input(self, input_text: str) -> str:

        tasks = [
            self.analyze_emotions(input_text),
            self.analyze_intent(input_text),
            self.extract_key_content(input_text),
            self.memo_ai(input_text)
        ]

        # ml_results = self.machine_analyze_emotion(input_text)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        emotions, intent, key_context,memo_ai = [result for result in results if result is not None]
        emotions = emotions if isinstance(emotions, str) else "Error"
        intent = intent if isinstance(intent, str) else "Error"
        key_context = key_context if isinstance(key_context, str) else "Error"
        response = f"Emotion: {emotions}\nIntent: {intent}\nKey Context: {key_context}\nMemoAi：{memo_ai}"
        # response = f"机器学习预测结果：{ml_results}"
        return response
