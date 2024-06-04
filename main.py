# main.py
import asyncio
import hashlib
import json
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import AsyncContextManager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status

from langchain.text_splitter import CharacterTextSplitter


from langchain_community.embeddings import OllamaEmbeddings, ModelScopeEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.tongyi import Tongyi
from langchain_community.vectorstores import Milvus, Chroma

import os

from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, scoped_session
from sse_starlette import EventSourceResponse
from starlette.responses import JSONResponse, FileResponse

from ai.prompts.base_character import BASE_CHARACTER_PROMPT
from ai.prompts.fast_character import FAST_CHARACTER_PROMPT
from app.api.models import ChatRequest, WriteDiary, EventRequest, GameUser, RoleLog, GameUser, GenerateSound
from app.core import CharacterAgent
from langchain_community.document_loaders import DirectoryLoader

from app.core.tools.dialogue_tool import EmotionCompanionTool, FactTransformTool, ExpressionTool, InformationTool, \
    OpinionTool, DefenseTool, RepeatTool, Memory_SearchTool
from app.service.services import get_user_database, get_message_memory, get_entity_memory, DBContext
from data.database.mysql.entity_memory import EntityMemory
from data.database.mysql.message_memory import MessageMemory

from data.database.mysql.models import Message
from data.database.mysql.user_management import UserDatabase

from utils.placeholder_replacer import PlaceholderReplacer
from gradio_client import Client



# setup_database(engine)
# # 创建全局实例
# user_database = UserDatabase(Session)
# message_memory = MessageMemory(Session)
# entity_memory = EntityMemory(Session)




load_dotenv()


tongyi_api_key = os.getenv('DASHSCOPE_API_KEY')

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

# fast_llm = Ollama(model="qwen:32b",temperature=0.7, top_k=100,top_p=0.9,base_url="http://182.254.242.30:11434")
fast_llm = Tongyi(model_name="qwen-max", dashscope_api_key=tongyi_api_key)
# chatLLM = ChatTongyi(
#     streaming=True,
# )


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
                            tools=tools
                            )

def get_client():
    base_url = os.getenv("TTS_API_URL")
    client = Client(base_url)
    try:
        yield client
    finally:
        client.close()

@app.post("/generate_sound")
async def generate_wav(request: GenerateSound,client = Depends(get_client)):
    first_sentence = request.text.split('.')[0].strip()  # 这里简单地以句号分割获取第一句，根据实际情况调整
    # 生成8位的唯一ID
    short_uuid = str(uuid.uuid4())[:8]
    # 组合文件名
    file_name = f"{first_sentence}_{short_uuid}.wav"

    try:
        result = client.predict(
            "ai/sounds/激动—好耶！《特尔克西的奇幻历险》出发咯！.wav",  # 使用临时文件路径
            "好耶！《特尔克西的奇幻历险》出发咯",  # 与音频对应的文本
            "Chinese",  # 语言选项
            request.text,
            "Chinese",  # 生成文本的语言
            "Slice once every 4 sentences",  # 文本切片选项
            5,  # top_k 参数
            1,  # top_p 参数
            1,  # 温度参数
            False,  # 是否启用无参考模式
            fn_index=3  # 目标功能的索引
        )
        audio_path = result
        if os.path.isfile(audio_path):
            logging.info(f"语音合成成功: {request.text}")
            return FileResponse(audio_path, media_type="audio/wav", filename=file_name)
        else:
            return {"error": "Generated audio file not found."}
    except Exception as e:
        logging.error(f"语音合成失败: {str(e)}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": f"语音合成失败: {str(e)}"})




@app.post("/create_game_user")
async def add_game_user(request: GameUser, user_db=Depends( get_user_database)):
    try:
        result = user_db.add_game_user(game_uid=request.game_uid, username=request.user_name, role_name=request.role_name)
        return JSONResponse(content={"message": result}, status_code=status.HTTP_201_CREATED)
    except Exception as e:
        logging.error(f"添加游戏用户失败: {str(e)}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": f"添加游戏用户失败: {str(e)}"})

@app.post("/update_game_user")
async def update_game_user(request: GameUser, user_db=Depends( get_user_database)):
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
async def add_role_log(request: RoleLog, user_db=Depends( get_user_database), message_memory=Depends(get_message_memory)):
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


async def chat_event_generator(uid, user_name, role_name, input_text, role_status: str ,db_context):
    try:
        async for response_chunk in tuji_agent.response(guid=uid, user_name=user_name, role_name=role_name,
                                                        input_text=input_text, role_status=role_status,
                                                        db_context=db_context):
            print(response_chunk, end="", flush=True)
            yield response_chunk

    except ValueError as ve:
        logging.error(f"生成聊天响应时出现Value错误: {ve}")
        yield f"处理请求时出错: {ve}"
    except ConnectionError as ce:
        logging.error(f"与聊天服务连接错误: {ce}")
        yield f"服务暂时不可用: {ce}"
    except Exception as e:
        logging.error(f"聊天事件生成器中出现意外错误: {e}")
        yield f"发生了意外错误: {e}"


def get_db_context(user_db: UserDatabase = Depends(get_user_database),
                   message_memory: MessageMemory = Depends(get_message_memory),
                   entity_memory: EntityMemory = Depends(get_entity_memory)) -> DBContext:
    return DBContext(user_db=user_db, message_memory=message_memory, entity_memory=entity_memory)
@app.post("/game/chat")
async def generate(request: ChatRequest, db_context: DBContext = Depends(get_db_context)):
    logging.info(f"收到游戏聊天请求，UID: {request.uid}。 输入: {request.input}")
    try:
        user = db_context.user_db.get_user_by_game_uid(request.uid)
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        uid = user.guid
        user_name = user.username
        role_name = user.role_name

        generator = chat_event_generator(uid, user_name, role_name, request.input, role_status=request.role_status,
                                         db_context=db_context)
        return EventSourceResponse(generator)

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"启动聊天会话失败: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="启动聊天会话失败")



def validate_date(date_string, format="%Y-%m-%d %H:%M:%S"):
    try:
        return datetime.strptime(date_string, format)
    except ValueError:
        raise ValueError("Incorrect date format, should be YYYY-MM-DD HH:MM:SS")
async def write_diary_event_generator(uid, user_name, role_name, date_range, db_context: DBContext):
    date_start, date_end = date_range
    async for response_chunk in tuji_agent.write_diary(guid=uid, user_name=user_name, role_name=role_name, date_start=date_start, date_end=date_end,db_context=db_context):
        yield response_chunk

@app.post("/game/write_diary")
async def write_diary(request: WriteDiary,db_context: DBContext = Depends(get_db_context)):
    logging.info(f"游戏端日记请求，uid:{request.uid}")
    user = db_context.user_db.get_user_by_game_uid(request.uid)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    uid = user.guid
    user_name = user.username
    role_name = user.role_name

    date_start = validate_date(request.date_start) if request.date_start else None
    date_end = validate_date(request.date_end) if request.date_end else None
    date_range = (date_start, date_end)

    return EventSourceResponse(
        write_diary_event_generator(uid, user_name, role_name, date_range, db_context=db_context))


async def event_generator(uid:str,user_name,role_name,llm:BaseLLM, event:str,db_context: DBContext):
    async for response_chunk in tuji_agent.event_response(guid=uid,user_name=user_name,role_name=role_name,llm=llm,event=event,db_context=db_context):
        yield response_chunk


async def get_qwen_max_llm() -> BaseLLM:
    return Tongyi(model_name="qwen-max", temperature=0.7, top_k=100, top_p=0.9, dashscope_api_key=tongyi_api_key)

@app.post("/game/event_response")
async def event_response(request: EventRequest,db_context: DBContext = Depends(get_db_context),llm: BaseLLM = Depends(get_qwen_max_llm)):
    user = db_context.user_db.get_user_by_game_uid(request.uid)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

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

    if request.need_response :
        return EventSourceResponse(event_generator(uid,user_name=user_name,role_name=role_name,llm=llm,event=event,db_context=db_context))

    else:
        db_context.message_memory.add_message(
            Message(user_guid=uid, type="system", role="系统事件", message=event,
                    generate_from="SystemEvent"))
        logging.info("System event recorded successfully.")
        return JSONResponse(content={"message": "系统事件记录成功"})

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
