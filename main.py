# main.py
import asyncio
import json
import logging
import re
from datetime import datetime
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status

from langchain.text_splitter import CharacterTextSplitter



from langchain_community.embeddings import OllamaEmbeddings, ModelScopeEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.tongyi import Tongyi
from langchain_community.vectorstores import Milvus, Chroma

import os

from langchain_core.language_models import BaseLLM
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, scoped_session
from sse_starlette import EventSourceResponse
from starlette.responses import JSONResponse

from ai.models.c_sql import SQLChatMessageHistory
from ai.models.system import SystemMessage


from ai.prompts.base_character import BASE_CHARACTER_PROMPT
from ai.prompts.fast_character import FAST_CHARACTER_PROMPT
from app.api.models import ChatRequest, WriteDiary, EventRequest, GameUser, RoleLog, GameUser
from app.core import CharacterAgent
from langchain_community.document_loaders import DirectoryLoader

from app.core.tools.dialogue_tool import EmotionCompanionTool, FactTransformTool, ExpressionTool, InformationTool, \
    OpinionTool, DefenseTool, RepeatTool, Memory_SearchTool
from data.database.mysql.database_config import setup_database
from data.database.mysql.entity_memory import EntityMemory
from data.database.mysql.message_memory import MessageMemory
from data.database.mysql.models import Message
from data.database.mysql.user_management import UserDatabase

from utils.placeholder_replacer import PlaceholderReplacer

load_dotenv()
# 数据库连接字符串，从环境变量或配置文件读取
DATABASE_CONNECTION_STRING = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_CONNECTION_STRING, echo=False)
Session = scoped_session(sessionmaker(bind=engine))
setup_database(engine)
# 创建全局实例
user_database = UserDatabase(Session)
message_memory = MessageMemory(Session)
entity_memory = EntityMemory(Session)

# 依赖函数
def get_user_database():
    return user_database

def get_message_memory():
    return message_memory

def get_entity_memory():
    return entity_memory



database_url = os.getenv('DATABASE_URL')
tongyi_api_key = os.getenv('TONGYI_API_KEY')

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

from langchain_community.llms import Ollama


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
llm = Tongyi(model_name="qwen-max", top_p=0.7, dashscope_api_key=tongyi_api_key)

#sk-c58751725c2942ff99ceaa4d315d89d2

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
    connection_string=database_url
)
# fast_llm = Ollama(model="qwen:32b",temperature=0.7, top_k=100,top_p=0.9,base_url="http://182.254.242.30:11434")
fast_llm = Tongyi(model_name="qwen-max", dashscope_api_key=tongyi_api_key)
# fast_llm_7b = Ollama(model="qwen:14b",temperature=0.5,base_url="http://182.254.242.30:11434")

# testuid = "98cf155b-d0f5-4129-ae2c-338f6587e74c"
# history_buffer = chat_message_history.buffer(guid=testuid)
# print(history_buffer)

# message_memory = MessageMemory(Session())
# messages = message_memory.get_messages(user_guid="896a7915-aeb4-4bb7-a597-0ddc369efc59", count=20)
# messages_str = "\n".join([f"Message ID: {msg.id}, Content: {msg.message}" for msg in messages])

base_info = replacer.replace_dict_placeholders(BASE_CHARACTER_PROMPT, config)
tuji_agent = CharacterAgent(base_info=base_info,
                            character_info=tuji_info,
                            llm=llm,
                            fast_llm=fast_llm,
                            retriever=retriever,
                            vector_db =vectordb,
                            tools=tools,
                            history=chat_message_history)


@app.post("/create_game_user")
async def add_game_user(request: GameUser, user_db=Depends(get_user_database)):
    try:
        result = user_db.add_game_user(game_uid=request.game_uid, username=request.user_name, role_name=request.role_name)
        return JSONResponse(content={"message": result}, status_code=status.HTTP_201_CREATED)
    except Exception as e:  # 捕获更广泛的异常，因为数据库层已处理SQLAlchemyError
        logging.error(f"添加游戏用户失败: {str(e)}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": f"添加游戏用户失败: {str(e)}"})

@app.post("/update_game_user")
async def update_game_user(request: GameUser, user_db=Depends(get_user_database)):
    try:
        result = user_db.update_game_user(game_uid=request.game_uid,
                                          new_user_name=request.user_name,
                                          new_role_name=request.role_name)
        if result:
            return JSONResponse(content={"message": result})
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户未找到.")
    except Exception as e:
        logging.error(f"更新游戏用户失败: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"更新游戏用户失败: {str(e)}")


@app.post("/game/add_role_log")
async def add_role_log(request: RoleLog, user_db=Depends(get_user_database), message_memory=Depends(get_message_memory)):
    try:
        # Fetch the user using their game UID
        user = user_db.get_user_by_game_uid(request.uid)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
        create_at = request.create_at
        # Attempt to log the message
        message_memory.add_message(
            message=Message(user_guid=user.guid,type="system",role="系统事件", message=request.log, created_at=create_at)
        )

        return JSONResponse(content={"message": "系统事件记录成功"}, status_code=status.HTTP_201_CREATED)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logging.error(f"添加系统事件日志异常: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"系统错误: {str(e)}")


async def chat_event_generator(uid, user_name, role_name, input_text, role_status: str = None):
    try:
        async for response_chunk in tuji_agent.response(guid=uid, user_name=user_name, role_name=role_name, input_text=input_text, role_status=role_status):
            print(response_chunk, end="", flush=True)
            yield response_chunk
    except ValueError as ve:
        logging.error(f"Value error in generating chat response: {ve}")
        yield f"Error in processing request: {ve}"
    except ConnectionError as ce:
        logging.error(f"Connection error with chat service: {ce}")
        yield "Service temporarily unavailable"
    except Exception as e:
        logging.error(f"Unexpected error in chat event generator: {e}")
        yield "An unexpected error occurred"

@app.post("/game/chat")
async def generate(request: ChatRequest, user_db: UserDatabase = Depends(get_user_database)):
    logging.info(f"Game chat request received, UID: {request.uid}. Input: {request.input}")
    try:
        user = user_db.get_user_by_game_uid(request.uid)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        uid = user.guid
        user_name = user.username
        role_name = user.role_name

        generator = chat_event_generator(uid, user_name, role_name, request.input, role_status=request.role_status)
        return EventSourceResponse(generator, media_type="text/event-stream")
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Failed to initiate chat session: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to initiate chat session")

async def write_diary_event_generator(uid, user_name, role_name, date_range):
    date_start, date_end = date_range
    async for response_chunk in tuji_agent.write_diary(guid=uid, user_name=user_name, role_name=role_name, date_start=date_start, date_end=date_end):
        yield response_chunk

@app.post("/game/write_diary")
async def write_diary(request: WriteDiary):
    logging.info(f"游戏端日记请求，uid:{request.uid}")
    user = user_database.get_user_by_game_uid(request.uid)
    uid = user.guid
    user_name = user.username
    role_name = user.role_name
    date_start = datetime.strptime(request.date_start, "%Y-%m-%d %H:%M:%S") if request.date_start else None
    date_end = datetime.strptime(request.date_end, "%Y-%m-%d %H:%M:%S") if request.date_end else None
    date_range = (date_start, date_end)
    return EventSourceResponse(write_diary_event_generator(uid, user_name, role_name, date_range))

# @app.post("/write_diary")
# async def write_diary (request: WriteDiary):
#     return EventSourceResponse(write_diary_event_generator(request.uid, request.date))

# @app.post("/login_event")
# async def login_event(request: ChatRequest):
#     return EventSourceResponse(chat_event_generator(request.uid, request.input))

async def event_generator(uid:str,user_name,role_name,llm:BaseLLM, event:str):
    async for response_chunk in tuji_agent.event_response(guid=uid,user_name=user_name,role_name=role_name,llm=llm,event=event):
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
    user = user_database.get_user_by_game_uid(request.uid)
    uid = user.guid
    user_name = user.username
    role_name = user.role_name
    event = (
        f"角色状态：{request.role_status}，"
        f"事件: {request.event_name},"
        f"发生时间：{request.create_at}，"
        f"发生地点：{request.event_location}，"
        f"事件详情：{request.event_description}，"
        f"事件反馈：{request.event_feedback}，"
        f"预期角色反应：{request.anticipatory_reaction}")
    # llm = Ollama(model="qwen:32b", temperature=0.7, top_k=100,top_p=0.9,base_url="http://182.254.242.30:11434")
    llm = Tongyi(model_name="qwen-max", temperature=0.7, top_k=100,top_p=0.9, dashscope_api_key=tongyi_api_key)
    if request.need_response :
        return EventSourceResponse(event_generator(uid,user_name=user_name,role_name=role_name,llm=llm,event=event))

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
