# main.py
import asyncio
import json
import logging
import re
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI

from langchain.text_splitter import CharacterTextSplitter



from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.llms.tongyi import Tongyi
from langchain_community.vectorstores.chroma import Chroma

import os


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sse_starlette import EventSourceResponse




from ai.models.c_sql import SQLChatMessageHistory

from ai.models.user import UserDatabase
from ai.prompts.base_character import BASE_CHARACTER_PROMPT
from ai.prompts.fast_character import FAST_CHARACTER_PROMPT
from app.api.models import ChatRequest, WriteDiary, EventRequest
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




# 加载JSON配置文件
with open('ai/prompts/character/tuji.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

replacer = PlaceholderReplacer()
# 使用函数替换占位符，生成填充后的提示字符串
tuji_info = replacer.replace_dict_placeholders(FAST_CHARACTER_PROMPT, config)
print(tuji_info)

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

embedding_model = "thenlper/gte-small-zh"
# 创建嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': "cpu"},
                                                encode_kwargs={'normalize_embeddings': True})

# 构建向量数据库
vectordb = Chroma.from_documents(documents=docs, embedding=embedding_model)


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
    session_id="test_session",
    connection_string="mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent",
)

history_buffer = chat_message_history.buffer()

base_info = replacer.replace_dict_placeholders(BASE_CHARACTER_PROMPT, config)
tuji_agent = CharacterAgent(base_info=base_info,character_info=tuji_info, llm=llm, retriever=retriever,vector_db =vectordb,tools=tools,history=chat_message_history)

testuid = "98cf155b-d0f5-4129-ae2c-338f6587e74c"

async def chat_event_generator(uid, input_text):

    async for response_chunk in tuji_agent.response(uid=uid, input_text=input_text):
        yield f"data: {response_chunk}\n\n"

@app.post("/chat")
async def generate(request: ChatRequest):
    return EventSourceResponse(chat_event_generator(request.uid, request.input))


async def write_diary_event_generator(uid, date):
    async for response_chunk in tuji_agent.write_diary(uid=uid,date=date):
        yield f"data: {response_chunk}\n\n"


@app.post("/write_diary")
async def write_diary (request: WriteDiary):
    return EventSourceResponse(write_diary_event_generator(request.uid, request.date))

# @app.post("/login_event")
# async def login_event(request: ChatRequest):
#     return EventSourceResponse(chat_event_generator(request.uid, request.input))

async def event_generator(uid, event):
    async for response_chunk in tuji_agent.event_response(uid=uid,event=event):
        yield f"data: {response_chunk}\n\n"


@app.post("/event_response")
async def event_response(request: EventRequest):
    event = (f"事件来源: {request.event_from},"
             f"动作：{request.action}，"
             f"动作对象：{request.action_object}，"
             f"对象描述：{request.object_description}，"
             f"对象反馈：{request.object_feedback}，"
             f"角色状态：{request.role_statu}，"
             f"预期角色反应：{request.anticipatory_reaction}")

    return EventSourceResponse(event_generator(request.uid, event))


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
