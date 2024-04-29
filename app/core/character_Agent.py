from typing import Any, Dict, List, AsyncGenerator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from app.core.abstract_Agent import AbstractAgent


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
        self.similarity_threshold = 0.5

        # Setup chains
        self.setup_and_retrieval = RunnableParallel({"classic_scenes": retriever, "input": RunnablePassthrough()})
        self.fast_chain = self.setup_and_retrieval | self.prompt_template | self.llm | self.output_parser

        self.deep_chain_template = PromptTemplate.from_template(
            "使用思维连一步一步回到下面的问题:\n\nQuestion: {input}\nAnswer:")
        self.deep_chain = self.deep_chain_template | self.llm


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

    async def response(self, prompt_text):
        retriever_lambda = RunnableLambda(self.rute_retriever)
        retriever_chain = retriever_lambda

        async for chunk in retriever_chain.astream(prompt_text):
            print(chunk, end="|", flush=True)

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


