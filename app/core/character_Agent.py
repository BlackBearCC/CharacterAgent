import json
from typing import Any, Dict, List, AsyncGenerator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from ai.prompts.deep_character import DEEP_CHARACTER_PROMPT
from app.core.abstract_Agent import AbstractAgent
from app.core.tools.dialogue_tool import EmotionCompanionTool, FactTransformTool, DialogueTool
from utils.placeholder_replacer import PlaceholderReplacer

import logging
class CharacterAgent(AbstractAgent):

    def __init__(self, character_info: str,retriever, document_util, llm,tools):
        self.character_info = character_info
        self.llm = llm
        self.memory = {}
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.tools = tools


        self.prompt_template = PromptTemplate(template=character_info, input_variables=["classic_scenes", "input"])
        self.output_parser = StrOutputParser()
        self.retriever = retriever
        self.document_util = document_util
        self.llm = llm
        self.similarity_threshold = 0.1

        # Setup chains
        self.setup_and_retrieval = RunnableParallel({"classic_scenes": retriever, "input": RunnablePassthrough()})
        self.fast_chain = self.setup_and_retrieval | self.prompt_template | self.llm | self.output_parser

        # 加载JSON配置文件
        with open('../ai/prompts/character/tuji.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        replacer = PlaceholderReplacer()

        # 替换配置占位符
        tuji_info = replacer.replace_dict_placeholders(DEEP_CHARACTER_PROMPT, config)

        # 替换工具占位符

        final_prompt = replacer.replace_tools_with_details(tuji_info,tools)
        logging.info("==============替换工具后的提示字符串===============\n"+final_prompt)

        self.deep_prompt_template = PromptTemplate(template=final_prompt, input_variables=["input"])

        self.deep_chain = self.deep_prompt_template | self.llm | self.output_parser

        self.chain_mapping = {}

        # 将列表转换为字典
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def set_chain_mapping(self, new_mapping):
        """设置新的 chain_mapping"""
        self.chain_mapping = new_mapping

    async def rute_retriever(self, query):
        docs_and_scores = self.document_util.vectordb.similarity_search_with_score(query)
        print(docs_and_scores)

        scores = [score for _, score in docs_and_scores]
        avg_score = sum(scores) / len(scores) if scores else 0

        if avg_score < self.similarity_threshold:
            print("==============相似度分数低于阈值，使用原始prompt和检索到的文档===============")
            return self.fast_chain
        else:
            print("==============相似度分数高于阈值，使用深度思考Agent===============")
            docs = [doc for doc, _ in docs_and_scores]
            # Although docs are calculated, they are not directly used here as per the original logic.
            return self.deep_chain

    def use_tool_by_name(self, action_name: str, action_input: str) -> Any:
        """
        根据工具名称调用对应工具的方法，并传入action_input。

        :param action_name: 要调用的工具的名称。
        :param action_input: 传给工具方法的输入字符串。
        :return: 工具方法执行后的返回值。
        """



        logging.info(f"尝试根据名称 '{action_name}' 调用工具...")
        for tool_name, tool_instance in self.tools_dict.items():
            # 假设每个工具类有一个属性 'name' 来标识它
            if hasattr(tool_instance, 'name') and tool_instance.name == action_name:
                logging.info(f"找到工具 '{tool_name}', 准备调用其方法...")

                if hasattr(tool_instance, 'strategy'):
                    response = tool_instance.strategy(action_input)
                    logging.info(f"工具 '{tool_name}' 处理完成。")
                    return response

                else:
                    logging.warning(f"工具 '{tool_name}' 缺少预期的处理方法。")
                    break
        else:
            logging.warning(f"未找到名为 '{action_name}' 的工具。")

        return None  # 如果没有找到匹配的工具或方法，则返回None或其他默认值

    async def route_post_deep_chain(self, deep_chain_output):
        """
        根据 deep_chain_output 决定使用哪一个链。

        参数:
            deep_chain_output: 一个字典，期望包含键 'action'，其值指示应使用的链。

        返回:
            字符串，表示选定的链，如果没有匹配的链，则返回 None。
        """

        try:
            action_name = deep_chain_output.get("action")
            if action_name is None:
                logging.info("action_name 为空,无策略调用")
                return None

            # 验证 action_name 是否为字符串类型
            if not isinstance(action_name, str):
                logging.error("action_name 非字符串类型")
                return None


            logging.info("Agent Use Chain: %s", action_name)
            self.use_tool_by_name(action_name=action_name,action_input="你好啊啊啊啊")



            # if action_name in self.tools:
            #     logging.info("Agent Use Chain: %s", action_name)

            # selected_chain = self.chain_mapping.get(action_name)
            # if selected_chain:
            #     logging.info(selected_chain)
            #     return selected_chain
            # else:
            #     logging.info("No matching chain found.")
            #
            # return None

        except Exception as e:
            logging.error("处理 route_post_deep_chain 时发生异常: %s", str(e))
            return None
    async def response(self, prompt_text):
        retriever_lambda = RunnableLambda(self.rute_retriever)
        retriever_chain = retriever_lambda
        output = ""
        async for chunk in retriever_chain.astream(prompt_text):
            # print(chunk, flush=True)
            output += chunk
            print("output:", output)
        try:
            # 将输出解析为 JSON 对象
            json_output = json.loads(output)
            print("\nValid JSON output:", json_output)
            await self.route_post_deep_chain(json_output)
        except json.JSONDecodeError:
            print("\nInvalid JSON output")


        #
        # # 检查 JSON 对象中是否包含指定的关键字
        # if "story" in json_output.get("result", "").lower():
        #     post_deep_chain_output = await self.story_chain.arun(json_output)
        #     print(f"\nPost Deep Chain Output: {post_deep_chain_output}")
        # elif "poem" in json_output.get("result", "").lower():
        #     post_deep_chain_output = await self.poem_chain.arun(json_output)
        #     print(f"\nPost Deep Chain Output: {post_deep_chain_output}")
        # else:
        #     print("\nNo additional chain to run.")



    # async def response_stream(self,input_text: str):
    #     async for chunk in self.model.astream_with_langchain(input_text):
    #         print(chunk, end="|", flush=True)
    #
    # async def response_stream_with_retriever(self,input_text: str, retriever):
    #     async for chunk in self.model.astream_with_langchain_RAG(retriever,input_text):
    #         print(chunk, end="|", flush=True)
    def perform_task(self, task: str, data: dict) -> int:
        return 200

    def remember(self, key: str):
        return 200

    def recall(self, key: str) -> any:
        pass


