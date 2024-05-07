# main.py
import asyncio
import json
import re

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
from ai.models.embedding.re_HuggingFaceBgeEmbeddings import ReHuggingFaceBgeEmbeddings
from ai.prompts.fast_character import FAST_CHARACTER_PROMPT
from app.core import CharacterAgent
from langchain_community.document_loaders import DirectoryLoader

from app.core.tools.dialogue_tool import EmotionCompanionTool, FactTransformTool, ExpressionTool, InformationTool, \
    OpinionTool, DefenseTool, RepeatTool, TopicTool
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
llm = Tongyi(model_name="qwen-turbo", top_p=0.7, dashscope_api_key="sk-dc356b8ca42c41788717c007f49e134a")




document_util = DocumentProcessingTool("../ai/knowledge/conversation_sample", chunk_size=100, chunk_overlap=20)
retriever = document_util.process_and_build_vector_db()

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
tuji_agent = CharacterAgent(character_info=tuji_info, llm=llm, retriever=retriever, document_util=document_util,tools=tools)


# # 定义角色描述
# role = "你是一位名叫Alice的AI助手,扮演一位善良、乐于助人的女仆角色。你必须坚持角色设定,不能违背女仆的身份和性格特征。"



# # 定义工具描述
# tools_desc = """现在，你已经有了一个角色，接下来，你需要用该角色的思维方式，按以下流程调用工具。
# ##回复流程
# EXTRACT：提取上下文关键信息。
# THOUGHT：分析并结合当前情境、状态、事件、记忆，优先考虑角色需求与情绪，并综合考虑用户状态与外部信息。
# ACTION：选择合适对话策略，should be one of [{tool_names}]\n\
#
#
#
# {tools}
# Thought: {agent_scratchpad}
# """
#
# tools_desc += "\n\n".join([f"{tool.name}: {tool.description}" for tool in tool_instances])

# 定义如何调用工具
tool_usage = """
##直接使用json格式输出你调用的策略和传给该策略的关键信息词组，不输出任何其他内容和完整自然语言。
使用以下格式来调用工具:
```json
{{"action": "tool_name", "action_input": "..."}}
如果你不需要工具,直接给出最终答案:
{{"action": "final_answer", "action_input": "..."}}

Example:
输入：我心情不好
输出：
{{
    "action": "情感陪伴",
    "action_input": "积极/开心/遇到事情"// Only reply with keywords, do not provide a full answer
}}
End.

开始：
"""

# system_message = f"{role}\n\n{tools_desc}\n\n{tool_usage}"
# system_prompt = SystemMessagePromptTemplate.from_template(system_message)
# ##定义人类输入提示模板
# human_template = HumanMessagePromptTemplate.from_template("{input}")


# prompt = ChatPromptTemplate.from_messages([system_prompt, human_template])

# memory = ConversationBufferMemory()




async def main():
    # await tuji_agent.response_stream(prompt_text)
    # await tuji_agent.response_stream_with_retriever(prompt_text,retriever)

    # chain.invoke("我们在哪")
    # print(chain.invoke("我们在哪"))

   await tuji_agent.response("你的冰箱有什么")


    # async for chunk in chain.astream("你好啊？"):
    #
    #     print(chunk, end="|", flush=True)


asyncio.run(main())
@app.get("/")
def read_root():
    return {"Hello": "World"}
