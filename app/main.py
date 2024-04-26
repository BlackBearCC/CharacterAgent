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

from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from ai.models import QianwenModel
from ai.models.embedding.re_HuggingFaceBgeEmbeddings import ReHuggingFaceBgeEmbeddings
from ai.prompts.fast_character import FAST_CHARACTER_PROMPT
from app.core import CharacterAgent
from langchain_community.document_loaders import DirectoryLoader

app = FastAPI()


model = QianwenModel()

tuji_agent = CharacterAgent("兔叽",model)
prompt_text = "我们在哪"

file_path = "../ai/knowledge/conversation_sample"

if os.path.exists(file_path):
    print(f"{file_path} exists.")
else:
    print(f"{file_path} does not exist.")

loader = DirectoryLoader(file_path,glob="**/*.txt",show_progress=True,use_multithreading=True)
documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=80, chunk_overlap=6)
docs = text_splitter.split_documents(documents)


model_name = "thenlper/gte-small-zh"  # 阿里TGE
# model_name = "BAAI/bge-small-zh-v1.5" # 清华BGE
encode_kwargs = {'normalize_embeddings': True}
embedding_model = ReHuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs=encode_kwargs
)
vectordb = Chroma.from_documents(documents=docs, embedding=embedding_model)
retriever = vectordb.as_retriever()

input_text = "我们在哪"
docs = vectordb.similarity_search(query=input_text,k=1)



"""
该脚本通过遍历文档集合，为每个文档生成格式化后的页面内容，并将其收集到一个字符串中。

文档集合应该包含具有页面内容格式化方法的文档对象。

参数:
- documents: 包含多个文档对象的集合，每个文档对象应有一个`page_content`属性，该属性是一个字符串格式模板。

返回值:
- 无。该脚本直接将生成的全部页面内容打印出来。
"""

# 初始化用于存储每个文档页面内容的列表
page_contents = []
for doc in documents:
    # 使用特定字符和用户名对文档的页面内容进行格式化
    content = doc.page_content.format(char="兔叽", user="哥哥")
    # 将格式化后的页面内容添加到列表中，每个页面内容后面添加一个换行符
    page_contents.append(content + "\n")

# 将页面内容列表转换为一个连续的字符串
page_contents_str = "".join(page_contents)
# 打印全部页面内容

def replace_dict_placeholders(prompt_string: str, config_dict: dict) -> str:
    """
    将字符串中的占位符{{key}}替换为配置字典中对应的值。如果key不存在，则保持原样返回{{key}}。

    参数:
    - prompt_string: 原始字符串，可能包含占位符。
    - config_dict: 包含键值对的字典，用于替换占位符。

    返回:
    - 替换后的字符串。
    """
    def replace(match):
        # 提取匹配的占位符key，并尝试从配置字典中获取其值
        key = match.group(1)
        return config_dict.get(key, f"{{{{{key}}}}}").format(**config_dict)

    # 定义正则表达式，匹配{{key}}形式的占位符
    pattern = re.compile(r"\{\{(.+?)\}\}")
    # 使用正则表达式和替换函数，替换所有占位符
    return pattern.sub(replace, prompt_string)

# 加载JSON配置文件
with open('../ai/prompts/character/tuji.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 使用函数替换占位符，生成填充后的提示字符串
filled_prompt = replace_dict_placeholders(FAST_CHARACTER_PROMPT, config)
print(filled_prompt)

# 初始化通义模型
llm = Tongyi(model_name="qwen-turbo", top_p=0.2, dashscope_api_key="sk-dc356b8ca42c41788717c007f49e134a")

# 准备提示模板和输入变量
prompt = PromptTemplate(template=filled_prompt,
                        input_variables=[ "classic_scenes","input"])
# 解析输出的parser
output_parser = StrOutputParser()
# 设置检索和准备数据的流程
setup_and_retrieval = RunnableParallel(
    {"classic_scenes": retriever, "input": RunnablePassthrough()}
)

# 构建处理链
chain = setup_and_retrieval|prompt | llm | output_parser


async def process_response(text, session_id, query):
    pass
async def main():
    # await tuji_agent.response_stream(prompt_text)
    # await tuji_agent.response_stream_with_retriever(prompt_text,retriever)

    # chain.invoke("我们在哪")
    # print(chain.invoke("我们在哪"))
    async for chunk in chain.astream("你好啊？"):
        print(chunk, end="|", flush=True)


asyncio.run(main())
@app.get("/")
def read_root():
    return {"Hello": "World"}
