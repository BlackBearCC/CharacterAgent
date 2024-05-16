import inspect
import json
from typing import Any, Dict, List, AsyncGenerator

from langchain.memory import ConversationBufferMemory, ConversationStringBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from ai.models.buffer import get_prefixed_buffer_string
from ai.models.c_sql import SQLChatMessageHistory
from ai.prompts.deep_character import DEEP_CHARACTER_PROMPT
from app.core.abstract_Agent import AbstractAgent
from utils.placeholder_replacer import PlaceholderReplacer

import logging
class CharacterAgent(AbstractAgent):

    def __init__(self, character_info: str,retriever, document_util, llm,tools,history:SQLChatMessageHistory):
        self.character_info = character_info
        self.llm = llm

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.tools = tools


        self.prompt_template = PromptTemplate(template=character_info, input_variables=["classic_scenes", "input"])
        self.output_parser = StrOutputParser()
        self.retriever = retriever
        self.document_util = document_util
        self.llm = llm
        self.similarity_threshold = 0.4

        # Setup chains
        self.setup_and_retrieval = RunnableParallel({"classic_scenes": retriever, "input": RunnablePassthrough()})
        self.fast_chain = self.setup_and_retrieval | self.prompt_template | self.llm | self.output_parser

        # 加载JSON配置文件
        with open('../ai/prompts/character/tuji.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.config = config

        self.history: SQLChatMessageHistory = history

        # 将列表转换为字典
        self.tools_dict = {tool.name: tool for tool in self.tools}

        self.user_input = ""







    async def rute_retriever(self, query):
        docs_and_scores = self.document_util.vectordb.similarity_search_with_score(query=query, k=3)
        print(docs_and_scores)

        scores = [score for _, score in docs_and_scores]
        avg_score = sum(scores) / len(scores) if scores else 0

        if avg_score < self.similarity_threshold:
            print("Agent : 相似度分数低于阈值，使用FastChain 进行回答")
            return self.fast_chain
        else:
            print("Agent : 相似度分数高于阈值，使用DeepChain 进行回答")
            docs = [doc for doc, _ in docs_and_scores]
            replacer = PlaceholderReplacer()
            # 替换配置占位符
            tuji_info = replacer.replace_dict_placeholders(DEEP_CHARACTER_PROMPT, self.config)
            # 替换历史占位符
            tuji_info_with_history = tuji_info.replace("{history}", self.history.buffer())
            # 替换工具占位符
            final_prompt = replacer.replace_tools_with_details(tuji_info_with_history, self.tools)
            logging.info("==============替换工具后的提示字符串===============\n" + final_prompt)

            deep_prompt_template = PromptTemplate(template=final_prompt, input_variables=["input"])

            deep_chain = deep_prompt_template | self.llm | self.output_parser
            return deep_chain

    async def use_tool_by_name(self, action_name: str, action_input: str) -> Any:
        """
        根据工具名称调用对应工具的方法，并传入action_input。

        :param action_name: 要调用的工具的名称。
        :param action_input: 传给工具方法的输入字符串。
        :return: 工具方法执行后的返回值。
        """
        # 日志记录尝试调用策略的开始
        logging.info(f"尝试根据名称 '{action_name}' 调用策略...")

        # 遍历所有工具，寻找与action_name匹配的工具
        for tool_name, tool_instance in self.tools_dict.items():

            # 检查工具实例是否具有名称属性且与action_name匹配
            if hasattr(tool_instance, 'name') and tool_instance.name == action_name:
                # 找到匹配的工具，准备调用其方法
                logging.info(f"找到策略 '{tool_name}', 准备调用其方法...")

                # 检查工具实例是否具有预期的处理方法
                # 处理异步生成器，注意需要strategy返回的是异步生成器而不是string，否则无法在外部流式返回网络请求结果
                if hasattr(tool_instance, 'strategy'):

                    # 根据策略方法的返回类型（异步生成器或协程），进行相应的处理
                    response_gen = tool_instance.strategy(user_input=self.user_input, action_input=action_input)
                    if inspect.isasyncgen(response_gen):  # 如果是异步生成器
                        return response_gen
                    else:
                        response = await response_gen  # 直接等待协程结果

                    # 记录策略处理完成
                    logging.info(f"策略 '{tool_name}' 处理完成。")
                    return response_gen

                else:
                    # 如果找到工具但缺少预期的方法，记录警告信息
                    logging.warning(f"策略 '{tool_name}' 缺少预期的处理方法。")
                    break
        else:
            # 如果遍历所有工具后仍未找到匹配的工具，记录警告信息
            logging.warning(f"未找到名为 '{action_name}' 的策略。")

        # 如果没有找到匹配的工具或方法，则返回None
        return None

    async def route_post_deep_chain(self, deep_chain_output):
        """
        根据 deep_chain_output 决定使用哪一个链。

        参数:
            deep_chain_output: 一个字典，期望包含键 'action'，其值指示应使用的链。

        返回:
            字符串，表示选定的链，如果没有匹配的链，则返回 None。
        """
        # 暂时写死，json格式，计划根据prompt动态处理
        action_name = deep_chain_output.get("action")
        action_input = deep_chain_output.get("input")

        if action_name is None:
            logging.info("Agent action_name 为空,无策略调用")
            return None

        # 验证 action_name 是否为字符串类型
        if not isinstance(action_name, str):
            logging.error("Agent action_name 非字符串类型")
            return None

        logging.info("Agent Use Chain: %s", action_name)
        return await self.use_tool_by_name(action_name=action_name, action_input=action_input)

    async def response(self, prompt_text: str):
        retriever_lambda = RunnableLambda(self.rute_retriever)
        retriever_chain = retriever_lambda
        final_output = ""
        self.user_input = prompt_text
        self.history.add_user_message(prompt_text)
        logging.info(f"User Input: {prompt_text}")
        logging.info("Agent : 检索对话知识库中...")

        async for chunk in retriever_chain.astream(prompt_text):
            final_output += chunk
            print(chunk, end="|", flush=True)

        def handle_output(output):
            try:
                json_output = json.loads(output)
                logging.info(f"Agent Action: {json_output}")
                self.history.add_ai_message(json_output["action"] + json_output["input"])
                return json_output
            except json.JSONDecodeError:
                logging.info("Agent Action: Use FastChain")
                return output

        final_json_output = handle_output(final_output)

        if isinstance(final_json_output, dict):
            thought_step = ""
            async for chunk in await self.route_post_deep_chain(final_json_output):
                thought_step += chunk
                print(f"策略响应: {chunk}", flush=True)
            logging.info(f"Agent Deep Chain Output: {thought_step}")
            self.history.add_ai_message(thought_step)
        else:
            logging.info(f"Agent Fast Chain Output: {final_output}")
            self.history.add_ai_message(final_output)
            pass

    def perform_task(self, task: str, data: dict) -> int:
        return 200

    def remember(self, key: str):
        return 200

    def recall(self, key: str) -> any:
        pass


