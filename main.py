# main.py
import asyncio
import json
import logging
import re
from datetime import datetime
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException

from langchain.text_splitter import CharacterTextSplitter



from langchain_community.embeddings import OllamaEmbeddings, ModelScopeEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.tongyi import Tongyi
from langchain_community.vectorstores import Milvus, Chroma

import os

from langchain_core.language_models import BaseLLM
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sse_starlette import EventSourceResponse
from starlette.responses import JSONResponse

from ai.models.c_sql import SQLChatMessageHistory
from ai.models.system import SystemMessage

from ai.models.user import UserDatabase
from ai.prompts.base_character import BASE_CHARACTER_PROMPT
from ai.prompts.fast_character import FAST_CHARACTER_PROMPT
from app.api.models import ChatRequest, WriteDiary, EventRequest, AddGameUser, RoleLog
from app.core import CharacterAgent
from langchain_community.document_loaders import DirectoryLoader

from app.core.tools.dialogue_tool import EmotionCompanionTool, FactTransformTool, ExpressionTool, InformationTool, \
    OpinionTool, DefenseTool, RepeatTool, Memory_SearchTool

from utils.placeholder_replacer import PlaceholderReplacer


# 创建数据库引擎
DATABASE_URL = "mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent"
engine = create_engine(DATABASE_URL)

# 创建Session工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
app = FastAPI()


# 创建一个日志处理器
class SSLFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()

        if 'SSLError' in message:
            return False
        return True

# 设置日志基本配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[RotatingFileHandler('app/app.log', maxBytes=10000, backupCount=5)]
)

# 配置requests的日志级别
requests_log = logging.getLogger("requests")
requests_log.setLevel(logging.CRITICAL)

# 创建一个过滤器，用于过滤特定的日志
handler = logging.root.handlers[0]
handler.addFilter(SSLFilter())

query = """Tell me a jokeapplication/x-ndjson (NDJSON 或 newline-delimited JSON) 是一种数据格式，用于在HTTP响应或流式传输中发送多个JSON对象。这种格式的特点是每个完整的JSON对象之间用换行符分隔，而不是像普通的JSON数组那样包含在一个大括号中。由于每个对象都是独立的，因此可以逐行解析和处理，而不需要等待整个响应完成。
当服务器以application/x-ndjson格式发送数据时，的确会按顺序逐行发送。这意味着客户端可以一边接收数据，一边处理每一行，而无需等待所有数据到达。这对于处理大量数据或者实时流数据非常有用，因为它允许数据的即时处理和较低的内存占用。
如果你遇到服务器似乎只在完全输出完才返回内容的情况，可能有以下几个原因："""
from langchain_community.llms import Ollama

print("fast_llm=================================================")
# fast_llm = Ollama(model="qwen:32b",temperature=0.5,base_url="http://182.254.242.30:11434")
# for chunks in fast_llm.stream(query):
#     print(chunks, end="",flush=True)
# 加载JSON配置文件
with open('ai/prompts/character/tuji.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

replacer = PlaceholderReplacer()
# 使用函数替换占位符，生成填充后的提示字符串
tuji_info = replacer.replace_dict_placeholders(FAST_CHARACTER_PROMPT, config)


# 初始化通义模型
llm = Tongyi(model_name="qwen-max", top_p=0.7, dashscope_api_key="sk-dc356b8ca42c41788717c007f49e134a")



file_path = "ai/knowledge/conversation_sample"

# 加载文档
loader = DirectoryLoader(file_path, glob="**/*.txt", show_progress=True,
                                 use_multithreading=True)
documents = loader.load()

# 分割文档为片段
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# embedding_model = "thenlper/gte-small-zh"
embedding_model = "iic/nlp_gte_sentence-embedding_chinese-small"
# embedding_model = "milkey/gte:large-zh-f16"
# 创建嵌入模型
# embedding_model = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': "cpu"},
#                                                 encode_kwargs={'normalize_embeddings': True})
embeddings = ModelScopeEmbeddings(model_id=embedding_model)
# embeddings = OllamaEmbeddings(base_url= "http://182.254.242.30:11434", model=embedding_model, temperature=0.5,)

# 构建向量数据库
vectordb = Chroma.from_documents(documents=docs, embedding=embeddings)

# vectordb = Milvus.from_documents(
#     docs,
#     embeddings,
#     collection_name="my_collection2",
#     connection_args={"host": "182.254.242.30", "port": "19530"},
#
# )
# vectordb = Milvus(
#     embedding_function=embeddings,
#     collection_name="my_collection",
#     auto_id=True,
#     connection_args={"host": "182.254.242.30", "port": "19530"},
# )
# # vectordb.delete()
# # 从 Document 对象中提取文本和元数据
# texts = [doc.page_content for doc in docs]
# metadatas = [doc.metadata for doc in docs]
#
# # vectordb.from_texts(texts=[doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs], embedding=embeddings, dimension=1024)
# # vectordb.add_texts("textdassssddddddddddddddddddddddddddds", dimension=1024)
# vectordb.add_texts(
#     texts=texts,
#     metadatas=metadatas,
#     dimension=512
# )
# document_util = DocumentProcessingTool("ai/knowledge/conversation_sample", chunk_size=100, chunk_overlap=20)
retriever = vectordb.as_retriever()

connection_string = "mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent"
user_database = UserDatabase(connection_string)
# user_database.add_user("test_user", "test_user@example.com")

tools = [
    EmotionCompanionTool(),  # 情感陪伴
    FactTransformTool(),  # 事实转换
    ExpressionTool(),  # 表达诉求
    InformationTool(),  # 信息查找
    OpinionTool(),  # 观点评价
    DefenseTool(),  # 防御对话
    RepeatTool(),  # 重复表达
    # TopicTool(),  # 话题激发(先不做)
]
chat_message_history = SQLChatMessageHistory(
    connection_string="mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent",
)
fast_llm = Ollama(model="qwen:14b",temperature=0.5,base_url="http://182.254.242.30:11434")
# fast_llm_7b = Ollama(model="qwen:14b",temperature=0.5,base_url="http://182.254.242.30:11434")

# testuid = "98cf155b-d0f5-4129-ae2c-338f6587e74c"
# history_buffer = chat_message_history.buffer(guid=testuid)
# print(history_buffer)

base_info = replacer.replace_dict_placeholders(BASE_CHARACTER_PROMPT, config)
tuji_agent = CharacterAgent(base_info=base_info,character_info=tuji_info, llm=llm,fast_llm=fast_llm, retriever=retriever,vector_db =vectordb,tools=tools,history=chat_message_history)





@app.post("/create_game_user")
async def add_game_user(request: AddGameUser):
    try:
        result = user_database.add_game_user(game_uid=request.game_uid, username=request.username, email=request.email)
        return JSONResponse(content={"message":result})
    except SQLAlchemyError as e:
        logging.error(f"添加游戏用户失败: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "添加用户失败."})


@app.post("/game/add_role_log")
async def add_role_log(request: RoleLog):
    try:
        if not request.uid or not request.log:
            raise HTTPException(status_code=400, detail="UID和LOG不能为空")

        user = user_database.get_user_by_game_uid(request.uid)
        if user is None:
            raise HTTPException(status_code=400, detail="用户不存在")

        # 检查create_at是否已提供并转换为datetime对象
        create_at = None
        if request.create_at:
            try:
                create_at = datetime.strptime(request.create_at, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="CREATE_AT必须是%Y-%m-%d %H:%M:%S格式"
                )

        # 使用create_at或None调用add_message_with_uid()
        chat_message_history.add_message_with_uid(
            user.guid,
            SystemMessage(content=request.log, created_at=create_at)
        )

        return JSONResponse(content={"message": "系统事件记录成功"})
    except Exception as e:
        logging.error(f"添加系统事件日志异常: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"系统错误: {str(e)}"})






async def chat_event_generator(uid, input_text):

    async for response_chunk in tuji_agent.response(guid=uid, input_text=input_text):
        print(response_chunk,end="",flush=True)
        yield response_chunk




@app.post("/chat")
async def generate(request: ChatRequest):
    logging.info(f"常规请求：{request.input}")
    return EventSourceResponse(chat_event_generator(request.uid, request.input))


@app.post("/game/chat")
async def generate(request: ChatRequest):
    logging.info(f"游戏端对话请求，uid:{request.uid}.输入:{request.input}")
    uid = user_database.get_user_by_game_uid(request.uid).guid

    return EventSourceResponse(chat_event_generator(uid, request.input))

async def write_diary_event_generator(uid, date):
    async for response_chunk in tuji_agent.write_diary(guid=uid,date=date):
        yield response_chunk


@app.post("/game/write_diary")
async def write_diary (request: WriteDiary):
    logging.info(f"游戏端日记请求，uid:{request.uid}")
    uid = user_database.get_user_by_game_uid(request.uid).guid
    return EventSourceResponse(write_diary_event_generator(uid, request.date))

@app.post("/write_diary")
async def write_diary (request: WriteDiary):
    return EventSourceResponse(write_diary_event_generator(request.uid, request.date))

# @app.post("/login_event")
# async def login_event(request: ChatRequest):
#     return EventSourceResponse(chat_event_generator(request.uid, request.input))

async def event_generator(uid:str,llm:BaseLLM, event:str):
    async for response_chunk in tuji_agent.event_response(guid=uid,llm=llm,event=event):
        yield response_chunk


@app.post("/chat_test")
async def generate(request: ChatRequest):

    url = 'http://182.254.242.30:11434/api/generate'
    data = {'model': 'qwen:14b', 'prompt': '为什么天空是蓝色，大象有鼻子'}
    fast_llm = Ollama(model="qwen:14b", temperature=0.5, base_url="http://182.254.242.30:11434")
    async def sse_generator():
        for chunks in fast_llm.stream(request.input):
            print(chunks,end="",flush=True)
            yield chunks

    # return EventSourceResponse(chat_event_generator(request.uid, request.input))
    return EventSourceResponse(sse_generator())
@app.post("/game/event_response")
async def event_response(request: EventRequest):
    uid = user_database.get_user_by_game_uid(request.uid).guid
    event = (
        f"角色状态：{request.role_status}，"
        f"事件: {request.event_name},"
        f"发生时间：{request.create_at}，"
        f"发生地点：{request.event_location}，"
        f"事件详情：{request.event_description}，"
        f"事件反馈：{request.event_feedback}，"
        f"预期角色反应：{request.anticipatory_reaction}")
    llm = Ollama(model="qwen:32b", temperature=0.7, top_k=100,top_p=0.9,base_url="http://182.254.242.30:11434")
    # llm = Tongyi(model_name="qwen-max", top_p=0.7, dashscope_api_key="sk-dc356b8ca42c41788717c007f49e134a")
    if request.need_response :
        return EventSourceResponse(event_generator(uid,llm=llm,event=event))
    else:
        chat_message_history.add_message_with_uid(guid=uid, message=SystemMessage(content=event))
        return JSONResponse(content={"message":"系统事件记录成功"})

# @app.post("/event_response")
# async def event_response(request: EventRequest):
#     event = (
#         f"角色状态：{request.role_status}，"
#         f"事件: {request.event_name},"
#         f"发生时间：{request.create_at}，"
#         f"发生地点：{request.event_location}，"
#         f"事件详情：{request.event_description}，"
#         f"事件反馈：{request.event_feedback}，"
#         f"预期角色反应：{request.anticipatory_reaction}")
#     if request.need_response :
#         return EventSourceResponse(event_generator(request.uid, event))
#     else:
#         chat_message_history.add_message_with_uid(guid=request.uid, message=SystemMessage(content=event))
#         return JSONResponse(content={"message":"系统事件记录成功"})



# async def main():
#
#     # 创建对话历史记录内存对象
#     memory = ConversationBufferMemory(human_prefix="大头哥", ai_prefix="兔几妹妹")
#     while True:
#         user_input = input("请输入你的消息：")
#         if user_input.lower() == "退出":
#             break
#         await tuji_agent.response(uid=testuid,input_text=user_input)
#
#
# asyncio.run(main())



@app.get("/")
def read_root():
    return {"Hello": "World"}
