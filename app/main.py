# main.py
import asyncio
import json
import re

from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms.tongyi import Tongyi
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

from ai.models import QianwenModel
from ai.models.embedding.re_HuggingFaceBgeEmbeddings import ReHuggingFaceBgeEmbeddings
from ai.prompts.fast_character import FAST_CHARACTER_PROMPT
from app.core import CharacterAgent
from langchain_community.document_loaders import DirectoryLoader

from utils.document_processing_tool import DocumentProcessingTool
from utils.placeholder_replacer import PlaceholderReplacer

app = FastAPI()


model = QianwenModel()

# 加载JSON配置文件
with open('../ai/prompts/character/tuji.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

replacer = PlaceholderReplacer()
# 使用函数替换占位符，生成填充后的提示字符串
tuji_info = replacer.replace_dict_placeholders(FAST_CHARACTER_PROMPT, config)
print(tuji_info)

tuji_agent = CharacterAgent(character_info=tuji_info, model=model)


document_util = DocumentProcessingTool("../ai/knowledge/conversation_sample", chunk_size=100, chunk_overlap=20)
retriever = document_util.process_and_build_vector_db()


# 初始化通义模型
llm = Tongyi(model_name="qwen-turbo", top_p=0.2, dashscope_api_key="sk-dc356b8ca42c41788717c007f49e134a")

# 准备提示模板和输入变量
prompt = PromptTemplate(template=tuji_info,
                        input_variables=[ "classic_scenes","input"])
# 解析输出的parser
output_parser = StrOutputParser()
# 设置检索和准备数据的流程
setup_and_retrieval = RunnableParallel(
    {"classic_scenes": retriever, "input": RunnablePassthrough()}
)
fast_chain = setup_and_retrieval | prompt | llm | output_parser

deep_chain = PromptTemplate.from_template(
    """使用思维连一步一步回到下面的问题:

Question: {input}
Answer:"""
) | llm

# 定义相似度阈值
similarity_threshold = 0.5
def rute_retriever(query):
    # 检索相关文档及其分数
    docs_and_scores = document_util.vectordb.similarity_search_with_score(query)

    print(docs_and_scores)

    # 计算平均相似度分数
    scores = [score for _, score in docs_and_scores]
    avg_score = sum(scores) / len(scores) if scores else 0

    if avg_score < similarity_threshold:
        print("==============相似度分数低于阈值，使用原始prompt和检索到的文档===============")
        return fast_chain
    else:
        # 否则使用原始prompt和检索到的文档
        print("==============相似度分数高于阈值，使用深度思考Agent===============")

        docs = [doc for doc, _ in docs_and_scores]
        return deep_chain

retriever = RunnableLambda(rute_retriever)

# 检索路由链
retriever_chain = retriever


async def process_response(text, session_id, query):
    pass
async def main():
    # await tuji_agent.response_stream(prompt_text)
    # await tuji_agent.response_stream_with_retriever(prompt_text,retriever)

    # chain.invoke("我们在哪")
    # print(chain.invoke("我们在哪"))

    async for chunk in retriever_chain.astream("我们在哪？"):

        print(chunk, end="|", flush=True)
    # async for chunk in chain.astream("你好啊？"):
    #
    #     print(chunk, end="|", flush=True)


asyncio.run(main())
@app.get("/")
def read_root():
    return {"Hello": "World"}
