import logging
import os

from langchain_community.llms.tongyi import Tongyi
from langchain_core.language_models import BaseLanguageModel, BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from openai import RateLimitError

from modules.data_context import DataContextManager


class BaseMindModule:
    def __init__(self, data_context_manager: DataContextManager):
        self.data_context_manager = data_context_manager
        self.tongyi_api_key = os.getenv('DASHSCOPE_API_KEY')

    async def invoke_chain(self, prompt_template, invoke_input: dict, **kwargs):
        llm = Tongyi(model_name="qwen-turbo", temperature=0.7, top_k=100, top_p=0.9, dashscope_api_key=self.tongyi_api_key)
        prompt_text = self.generate_prompt_text(prompt_template, **kwargs)
        prompt = PromptTemplate(template=prompt_text, input_variables=invoke_input.keys())
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        try:
            response = await chain.ainvoke(invoke_input)
            logging.info(f"Response: {response}")
            return response
        except RateLimitError:
            logging.error("Rate limit reached, retrying...")
            response = await self.retry_chain(chain, kwargs)
            return response
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")
            raise e

    def generate_prompt_text(self, prompt_template, **kwargs):
        for key, value in kwargs.items():
            prompt_template = prompt_template.replace(f'{{{key}}}', str(value))
        return prompt_template

    async def retry_chain(self, chain, kwargs):
        try:
            response = await chain.ainvoke(kwargs)
            logging.info(f"Retry Response: {response}")
            return response
        except Exception as e:
            logging.error(f"Retry failed: {e}")
            raise e
