import inspect
import json
from typing import Any, Dict, List, AsyncGenerator


from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough


from ai.models.buffer import get_prefixed_buffer_string


from ai.models.role_memory import OpinionMemory

from ai.prompts.deep_character import DEEP_CHARACTER_PROMPT
from ai.prompts.game_function import WRITE_DIARY_PROMPT, EVENT_PROMPT
from ai.prompts.reflexion import ENTITY_SUMMARIZATION_PROMPT
from app.core.abstract_Agent import AbstractAgent
from app.service.services import DBContext




from data.database.mysql.models import Message, Entity
from data.database.mysql.user_management import UserDatabase


from utils.placeholder_replacer import PlaceholderReplacer
from langchain_community.llms import Ollama



import logging
class CharacterAgent(AbstractAgent):

    def __init__(self,
                 base_info:str,character_info: str,vector_db,retriever, llm,fast_llm,tools):
        self.character_info = character_info
        self.llm = llm

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.tools = tools
        self.vector_db = vector_db
        self.fast_llm= fast_llm


        self.retriever = retriever

        self.llm = llm


        self.similarity_threshold = 0.4
        self.base_info = base_info


        # 加载JSON配置文件
        with open('ai/prompts/character/tuji.json', 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.config = config

        replacer = PlaceholderReplacer()

        # 替换配置占位符
        self.tuji_info = replacer.replace_dict_placeholders(DEEP_CHARACTER_PROMPT, self.config)




        # 将列表转换为字典
        self.tools_dict = {tool.name: tool for tool in self.tools}

        self.user_input = ""







    async def rute_retriever(self, guid:str,user_name,role_name, query: str,role_status:str, db_context: DBContext):
        logging.info("Agent : 检索对话知识库中...")
        docs_and_scores = self.vector_db.similarity_search_with_score(query=query, k=3)
        print(docs_and_scores)
        scores = [score for _, score in docs_and_scores]
        avg_score = sum(scores) / len(scores) if scores else 0
        print("平均相似度分数:", avg_score)
        entity = db_context.entity_memory.get_entity(guid)
        if entity is None:
            entity = Entity(entity="", summary="")
        print("实体："+entity.entity)
        opinion_memory = OpinionMemory(
            connection_string="mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent")
        role_state = "('体力':'饥饿','精力':'疲劳','位置':'房间，沙发上','动作':'坐着')"
        history = db_context.message_memory.buffer_messages(guid,user_name,role_name, 100)
        print("message_memory:"+history)

        if avg_score < self.similarity_threshold:
            print("Agent : 相似度分数低于阈值，使用FastChain 进行回答")
            # Setup chains
            info_with_state = self.character_info.replace("{role_state}", role_status)
            # 替换特殊记忆占位符
            info_with_special_memory = info_with_state.replace("{special_memory}", f"{entity.entity}:{entity.summary}")
            # 替换环境占位符
            info_with_environment = info_with_special_memory.replace("{environment}",
                                                                     "")
            info_with_opinion = info_with_environment.replace("{opinion}", opinion_memory.buffer(guid, 10))
            info_with_history = info_with_opinion.replace("{history}", history)
            # print("Agent FastChain 动态信息填充:", info_with_history)

            prompt_template = PromptTemplate(template=info_with_history, input_variables=["classic_scenes", "input"])
            output_parser = StrOutputParser()
            setup_and_retrieval = RunnableParallel({"classic_scenes": self.retriever, "input": RunnablePassthrough()})
            fast_chain = setup_and_retrieval | prompt_template | self.fast_llm | output_parser

            return fast_chain

        else:
            print("Agent : 相似度分数高于阈值，使用DeepChain 进行回答")
            docs = [doc for doc, _ in docs_and_scores]
            output_parser = StrOutputParser()
            # 替换特殊记忆占位符
            info_with_special_memory = self.tuji_info.replace("{special_memory}", f"{entity.entity}:{entity.summary}")

            # 替换环境占位符
            info_with_environment = info_with_special_memory.replace("{environment}", "")
            # 替换角色状态占位符
            info_with_state = info_with_environment.replace("{role_state}", role_status)
            # 替换观点占位符
            info_with_opinion =  info_with_state.replace("{opinion}",opinion_memory.buffer(guid,10) )
            # 替换历史占位符
            tuji_info_with_history = info_with_opinion.replace("{history}", history)
            # 替换工具占位符
            replacer = PlaceholderReplacer()
            final_prompt = replacer.replace_tools_with_details(tuji_info_with_history, self.tools)
            deep_prompt_template = PromptTemplate(template=final_prompt, input_variables=["input"])
            deep_chain = deep_prompt_template | self.llm | output_parser
            return deep_chain

    async def use_tool_by_name(self,guid:str,user_name,role_name,role_status:str, action_name: str, action_input: str, db_context: DBContext) -> Any:
        """
        根据工具名称调用对应工具的方法，并传入action_input。

        :param action_name: 要调用的工具的名称。
        :param action_input: 传给工具方法的输入字符串。
        :return: 工具方法执行后的返回值。
        """
        # 日志记录尝试调用策略的开始
        logging.info(f"尝试根据名称 '{action_name}' 调用策略...")

        # 遍历所有工具，寻找与action_name匹配的工具
        for tool_name, tool_instance in self.tools_dict.items():

            # 检查工具实例是否具有名称属性且与action_name匹配
            if hasattr(tool_instance, 'name') and tool_instance.name == action_name:
                # 找到匹配的工具，准备调用其方法
                logging.info(f"找到策略 '{tool_name}', 准备调用其方法...")

                # 检查工具实例是否具有预期的处理方法
                # 处理异步生成器，注意需要strategy返回的是异步生成器而不是string，否则无法在外部流式返回网络请求结果
                if hasattr(tool_instance, 'strategy'):

                    # 根据策略方法的返回类型（异步生成器或协程），进行相应的处理
                    response_gen = tool_instance.strategy(uid =guid,user_name=user_name,role_name=role_name,user_input=self.user_input, action_input=action_input, role_status=role_status,db_context=db_context)
                    if inspect.isasyncgen(response_gen):  # 如果是异步生成器
                        return response_gen
                    else:
                        response = await response_gen  # 直接等待协程结果

                    # 记录策略处理完成
                    logging.info(f"策略 '{tool_name}' 处理完成。")
                    return response_gen

                else:
                    # 如果找到工具但缺少预期的方法，记录警告信息
                    logging.warning(f"策略 '{tool_name}' 缺少预期的处理方法。")
                    break
        else:
            # 如果遍历所有工具后仍未找到匹配的工具，记录警告信息
            logging.warning(f"未找到名为 '{action_name}' 的策略。")

        # 如果没有找到匹配的工具或方法，则返回None
        return None

    async def route_post_deep_chain(self, guid:str,user_name,role_name,input,role_status:str,db_context:DBContext):
        """
        根据 deep_chain_output 决定使用哪一个链。

        参数:
            deep_chain_output: 一个字典，期望包含键 'action'，其值指示应使用的链。

        返回:
            字符串，表示选定的链，如果没有匹配的链，则返回 None。
        """
        # 暂时写死，json格式，计划根据prompt动态处理
        action_name = input.get("action")
        action_input = input.get("input")

        if action_name is None:
            logging.info("Agent action_name 为空,无策略调用")
            return None

        # 验证 action_name 是否为字符串类型
        if not isinstance(action_name, str):
            logging.error("Agent action_name 非字符串类型")
            return None

        logging.info("Agent Use Chain: %s", action_name)
        return await self.use_tool_by_name(guid=guid,user_name=user_name,role_name=role_name,action_name=action_name, action_input=action_input,role_status=role_status,db_context=db_context)

    async def response(self, guid:str ,user_name,role_name,input_text: str,role_status,db_context: DBContext) -> AsyncGenerator[str, None]:
        """
        异步处理用户输入，并生成相应的响应。

        :param uid:
        :param user_input: 用户的输入文本。
        :return: 无返回值，但会异步处理用户输入，并通过日志和历史记录对话过程。
        """

        # 初始化检索链
        # retriever_lambda = RunnableLambda(self.rute_retriever)
        # retriever_chain = retriever_lambda
        retriever_chain =await self.rute_retriever(guid=guid,user_name=user_name,role_name=role_name, query=input_text,role_status=role_status,db_context=db_context)
        step_message = ""
        final_output = ""  # 用于存储最终输出字符串
        self.user_input = input_text  # 存储用户输入

        human_message = Message(user_guid=guid, type="human", role=user_name, message=input_text,generate_from="GameUser")
        logging.info(f"{guid},User Input: {input_text}")  # 记录用户输入的日志
        db_context.message_memory.add_message(human_message)
        # self.history.add_message_with_uid(guid=guid,message=HumanMessage(content=input_text))
        # 通过检索链异步获取响应片段，并累加到最终输出
        async for chunk in retriever_chain.astream(input_text):
            final_output += chunk
            yield chunk
            # print(chunk, end="|", flush=True)

        def handle_output(output):
            """
            处理检索链的输出，尝试将其解析为JSON，失败则视为普通文本输出。

            :param output: 检索链的输出文本。
            :return: 解析后的JSON对象或原始文本。
            """
            try:
                json_output = json.loads(output)
                logging.info(f"Agent Action: {json_output}")
                # self.history.add_ai_message(message)
                return json_output
            except json.JSONDecodeError:
                logging.info("Agent Action: Use FastChain")
                return output

        final_json_output = handle_output(final_output)  # 处理最终的检索链输出

        if isinstance(final_json_output, dict):
            strategy_output = ""
            # 如果输出是字典，则进一步通过深度处理链处理，并累加响应
            async for chunk in await self.route_post_deep_chain(guid=guid,user_name=user_name,role_name=role_name, input=final_json_output,role_status=role_status,db_context=db_context):
                strategy_output += chunk
                # print(f"{chunk}", end="|", flush=True)
                yield chunk
            # logging.info(f"Agent Deep Chain Output: {strategy_output}")

            # 将字典转换为JSON格式的字符串
            # input_str = json.dumps(json_output["input"], ensure_ascii=False)
            concatenated_values = ''
            for key, value in final_json_output["input"].items():
                concatenated_values += f"{key}={value}" + ','
            step_message = f"Action: {final_json_output['action']} - Input: {concatenated_values}"

            ai_message = Message(user_guid=guid, type="ai", role=role_name, message=strategy_output,generate_from="Chat-DeepChain",call_step=step_message)
            db_context.message_memory.add_message(ai_message)
            logging.info(f"Agent Deep Chain Output: {strategy_output}")
            # self.history.add_message_with_uid(guid=guid,message=AIMessage(content=strategy_output,generate_from="DeepChain" , call_step=step_message))
        else:
            # 如果输出不是字典，则视为快速链输出
            logging.info(f"Agent Fast Chain Output: {final_output}")
            ai_message = Message(user_guid=guid, type="ai", role=role_name, message=final_output,
                                 generate_from="Chat-FastChain",call_step=step_message)
            db_context.message_memory.add_message(ai_message)
            logging.info(f"Agent Fast Chain Output: {final_output}")
            # self.history.add_message_with_uid(guid=guid,message=AIMessage(content=final_output,generate_from="FastChain" ,call_step=None))
            pass  # 忽略else块中的pass，避免修改原有代码逻辑

        # entity_memory = EntityMemory(
        #     connection_string="mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent")

        entity = db_context.entity_memory.get_entity(guid)
        output_parser = StrOutputParser()
        if entity is None:
            entity = Entity(entity=user_name,summary="",user_guid=guid)
        info_with_entity = ENTITY_SUMMARIZATION_PROMPT.replace("{entity}",entity.entity)
        entity_with_history = info_with_entity.replace("{history}",db_context.message_memory.buffer_messages(guid,user_name,role_name,count=100))
        entity_with_summary = entity_with_history.replace("{summary}",entity.summary)
        entity_prompt_template = PromptTemplate(template=entity_with_summary, input_variables=["input"],)
        reflexion_chain = entity_prompt_template | self.fast_llm | output_parser
        entity_output=""
        async for chunk in reflexion_chain.astream({"input":""}):
            entity_output += chunk
            print(f"{chunk}", end="|", flush=True)
        entity.summary = entity_output
        db_context.entity_memory.add_entity(entity)
        logging.info(f"Agent 实体更新: {entity}")
        #
    async def write_diary(self,user_name,role_name,guid:str,date_start,date_end,db_context:DBContext) -> AsyncGenerator[str, None]:
         # 替换配置占位符
        history_buffer = db_context.message_memory.buffer_messages(guid,user_name,role_name,count=100,start_date=date_start,end_date=date_end)
        info_with_role = WRITE_DIARY_PROMPT.replace("{role}",self.base_info)
        info_with_name = info_with_role.replace("{user}", user_name).replace("{char}", role_name)
        prompt_template = PromptTemplate(template=info_with_name, input_variables=[ "history"])
        output_parser = StrOutputParser()
        final_diary = ""
        diary_chain =  prompt_template | self.llm | output_parser
        async for chunk in diary_chain.astream({"history":history_buffer}):
            final_diary += chunk
            yield chunk
        logging.info(f"Agent Write Diary: {final_diary}")
        db_context.message_memory.add_message(Message(user_guid=guid, type="system", role="系统事件", message=f"{role_name}写了一篇日记：{final_diary}",
                                                        generate_from="WriteDiary"))
    async def event_response(self,user_name,role_name,llm:BaseLLM,guid:str,event: str,db_context:DBContext) -> AsyncGenerator[str, None]:
        info_with_role = EVENT_PROMPT.replace("{role}",self.base_info)
        info_with_history = info_with_role.replace("{history}",db_context.message_memory.buffer_messages(guid,user_name,role_name,count=30))
        info_name = info_with_history.replace("{user}", user_name)
        # print(info_name)
        prompt_template = PromptTemplate(template=info_name, input_variables=["event"])
        output_parser = StrOutputParser()
        event_chain = prompt_template | llm | output_parser
        results =""
        async for chunk in event_chain.astream({"event":event}):
            results+=chunk
            yield chunk
        system_message = Message(user_guid=guid, type="system", role="系统事件", message=event,
                             generate_from="SystemEvent")

        ai_message = Message(user_guid=guid, type="ai", role=role_name, message=results,
                             generate_from="SystemEvent")
        db_context.message_memory.add_messages([system_message,ai_message])
        logging.info(f"Agent System Event: {event}")
        logging.info(f"Agent System Event Response: {results}")

        # self.history.add_message_with_uid(guid=guid, message=SystemMessage(content=event))
        # self.history.add_message_with_uid(guid=guid, message=AIMessage(content=results,generate_from="Event"))





    def perform_task(self, task: str, data: dict) -> int:
        return 200

    def remember(self, key: str):
        return 200

    def recall(self, key: str) -> any:
        pass


