import json
from typing import Any, Dict, List, AsyncGenerator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from ai.prompts.deep_character import DEEP_CHARACTER_PROMPT
from app.core.abstract_Agent import AbstractAgent
from utils.placeholder_replacer import PlaceholderReplacer


class CharacterAgent(AbstractAgent):

    def __init__(self, character_info: str,retriever, document_util, llm):
        self.character_info = character_info
        self.llm = llm
        self.memory = {}


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
        # 使用函数替换占位符，生成填充后的提示字符串
        tuji_info = replacer.replace_dict_placeholders(DEEP_CHARACTER_PROMPT, config)
        print(tuji_info)
        self.deep_prompt_template = PromptTemplate(template=tuji_info, input_variables=[ "input"])

        self.deep_chain = self.deep_prompt_template | self.llm | self.output_parser


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

    async def route_post_deep_chain(self, deep_chain_output):
        # Implement your logic here to determine which chain to use
        # based on the deep_chain_output
        # For example:
        if "story" in deep_chain_output.get("result", ""):
            print("story")
            return self.story_chain
        elif "poem" in deep_chain_output.get("result", ""):
            print("poem")
            return self.poem_chain
        else:
            print("No matching chain found.")
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


