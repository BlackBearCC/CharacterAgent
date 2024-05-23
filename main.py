# main.py
import asyncio
import json
import logging
import re
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI
from langchain.agents import initialize_agent, AgentOutputParser, AgentExecutor, create_react_agent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import StructuredOutputParser
from langchain.text_splitter import CharacterTextSplitter


from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms.tongyi import Tongyi
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate
import os

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sse_starlette import EventSourceResponse
from starlette.middleware.sessions import SessionMiddleware


from ai.models.buffer import get_prefixed_buffer_string
from ai.models.c_sql import SQLChatMessageHistory
from ai.models.embedding.re_HuggingFaceBgeEmbeddings import ReHuggingFaceBgeEmbeddings
from ai.models.role_memory import OpinionMemory
from ai.models.user import UserDatabase
from ai.prompts.base_dialogue import BASE_STRATEGY_PROMPT
from ai.prompts.default_strategy import EMOTION_STRATEGY
from ai.prompts.fast_character import FAST_CHARACTER_PROMPT
from app.api.models import ChatRequest
from app.core import CharacterAgent
from langchain_community.document_loaders import DirectoryLoader

from app.core.tools.dialogue_tool import EmotionCompanionTool, FactTransformTool, ExpressionTool, InformationTool, \
    OpinionTool, DefenseTool, RepeatTool, Memory_SearchTool
from utils.document_processing_tool import DocumentProcessingTool
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


document_util = DocumentProcessingTool("ai/knowledge/conversation_sample", chunk_size=100, chunk_overlap=20)
retriever = document_util.process_and_build_vector_db()

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


tuji_agent = CharacterAgent(character_info=tuji_info, llm=llm, retriever=retriever, document_util=document_util,tools=tools,history=chat_message_history)

testuid = "98cf155b-d0f5-4129-ae2c-338f6587e74c"

async def chat_event_generator(uid, input_text):

    async for response_chunk in tuji_agent.response(uid=uid, input_text=input_text):
        yield f"data: {response_chunk}\n\n"

@app.post("/chat")
async def generate(request: ChatRequest):
    return EventSourceResponse(chat_event_generator(request.uid, request.input))

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
