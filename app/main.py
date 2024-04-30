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

# 初始化通义模型
llm = Tongyi(model_name="qwen-turbo", top_p=0.5, dashscope_api_key="sk-dc356b8ca42c41788717c007f49e134a")




document_util = DocumentProcessingTool("../ai/knowledge/conversation_sample", chunk_size=100, chunk_overlap=20)
retriever = document_util.process_and_build_vector_db()

tuji_agent = CharacterAgent(character_info=tuji_info, llm=llm, retriever=retriever, document_util=document_util)





async def main():
    # await tuji_agent.response_stream(prompt_text)
    # await tuji_agent.response_stream_with_retriever(prompt_text,retriever)

    # chain.invoke("我们在哪")
    # print(chain.invoke("我们在哪"))

   await tuji_agent.response("你好傻")


    # async for chunk in chain.astream("你好啊？"):
    #
    #     print(chunk, end="|", flush=True)


asyncio.run(main())
@app.get("/")
def read_root():
    return {"Hello": "World"}
