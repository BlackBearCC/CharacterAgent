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

from ai.models import QianwenModel
from ai.models.buffer import get_prefixed_buffer_string
from ai.models.c_sql import SQLChatMessageHistory
from ai.models.embedding.re_HuggingFaceBgeEmbeddings import ReHuggingFaceBgeEmbeddings
from ai.prompts.base_dialogue import BASE_STRATEGY_PROMPT
from ai.prompts.default_strategy import EMOTION_STRATEGY
from ai.prompts.fast_character import FAST_CHARACTER_PROMPT
from app.core import CharacterAgent
from langchain_community.document_loaders import DirectoryLoader

from app.core.tools.dialogue_tool import EmotionCompanionTool, FactTransformTool, ExpressionTool, InformationTool, \
    OpinionTool, DefenseTool, RepeatTool, TopicTool
from utils.document_processing_tool import DocumentProcessingTool
from utils.placeholder_replacer import PlaceholderReplacer


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
    handlers=[RotatingFileHandler('app.log', maxBytes=10000, backupCount=5)]
)

# 配置requests的日志级别
requests_log = logging.getLogger("requests")
requests_log.setLevel(logging.CRITICAL)

# 创建一个过滤器，用于过滤特定的日志
handler = logging.root.handlers[0]
handler.addFilter(SSLFilter())



model = QianwenModel()

# 加载JSON配置文件
with open('../ai/prompts/character/tuji.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

replacer = PlaceholderReplacer()
# 使用函数替换占位符，生成填充后的提示字符串
tuji_info = replacer.replace_dict_placeholders(FAST_CHARACTER_PROMPT, config)
print(tuji_info)

# 初始化通义模型
llm = Tongyi(model_name="qwen-max", top_p=0.7, dashscope_api_key="sk-dc356b8ca42c41788717c007f49e134a")




document_util = DocumentProcessingTool("../ai/knowledge/conversation_sample", chunk_size=100, chunk_overlap=20)
retriever = document_util.process_and_build_vector_db()

# base_strategy = replacer.replace_dict_placeholders(BASE_STRATEGY_PROMPT, config)
# base_strategy_template = base_strategy.replace("{answer_tendency}", EMOTION_STRATEGY)
# emotion_template = PromptTemplate(template=base_strategy_template, input_variables=[ "input"])
#
# logging.info(emotion_template)
# output_parser = StrOutputParser()
# emotion_chain = emotion_template | llm | output_parser


tools = [
    EmotionCompanionTool(),
    FactTransformTool(),
    ExpressionTool(),
    InformationTool(),
    OpinionTool(),
    DefenseTool(),
    RepeatTool(),
    TopicTool(),
]
chat_message_history = SQLChatMessageHistory(
    session_id="test_session",
    connection_string="mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent",
)
history_buffer = chat_message_history.buffer()

# history_buffer = get_prefixed_buffer_string(messages, "大头哥", "兔几妹妹")
# print(history_buffer)
tuji_agent = CharacterAgent(character_info=tuji_info, llm=llm, retriever=retriever, document_util=document_util,tools=tools,history_buffer=history_buffer)


async def main():


    # 创建对话历史记录内存对象
    memory = ConversationBufferMemory(human_prefix="大头哥", ai_prefix="兔几妹妹")
    while True:
        user_input = input("请输入你的消息：")
        if user_input.lower() == "退出":
            break
        await tuji_agent.response(user_input,chat_message_history)
        # message_strings = [str(message) for message in chat_message_history.messages(20)]
        # logging.info("当前对话历史记录：" + ", ".join(message_strings))
    # 获取消息列表


    # async for chunk in chain.astream("你好啊？"):
    #
    #     print(chunk, end="|", flush=True)


asyncio.run(main())
@app.get("/")
def read_root():
    return {"Hello": "World"}
