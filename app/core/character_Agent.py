import asyncio
import inspect
import json
from enum import Enum
from typing import Any, Dict, List, AsyncGenerator

from langchain_community.llms.tongyi import Tongyi
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ai.models.buffer import get_prefixed_buffer_string


from ai.models.role_memory import OpinionMemory

from ai.prompts.deep_character import DEEP_CHARACTER_PROMPT
from ai.prompts.game_function import WRITE_DIARY_PROMPT, EVENT_PROMPT
from ai.prompts.prompt_emum import PromptType
from ai.prompts.reflexion import ENTITY_SUMMARIZATION_PROMPT
from ai.prompts.task import EVENT_SUMMARY_TEMPLATE, BALDERDASH_TEMPLATE
from app.core.abstract_Agent import AbstractAgent
from app.core.tools.dialogue_tool import multiply
from app.core.tools.tools import emotion_companion, fact_conversion, information_lookup, express_needs, \
    opinion_evaluation, defensive_dialogue, repeat_expression
from app.service.services import DBContext




from data.database.mysql.models import Message, Entity, Message_Summary
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


        # self.similarity_threshold = 0.365
        self.similarity_threshold = 888
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

    async def response_fast(self, prompt_type: PromptType, db_context: DBContext, role_status, user_name, role_name, guid, query,
                            llm,remember:bool=True)->AsyncGenerator:
        """
        异步生成并流式返回快速聊天的响应。

        参数:
        - prompt_type: 提示类型，例如PromptType.FAST_CHAT。
        - db_context: 数据库上下文对象，用于获取消息历史。
        - role_status: 角色状态信息。
        - user_name: 用户名。
        - role_name: 角色名称。
        - guid: 消息唯一标识。
        - query: 用户的查询内容。
        - llm: 大型语言模型实例，用于生成回复。
        """
        # human_message = Message(user_guid=guid, type="human", role=user_name, message=query,
        #                         generate_from="GameUser")
        # logging.info(f"{guid},User Input: {query}")  # 记录用户输入的日志
        # db_context.message_memory.add_message(human_message)
        # 生成系统提示并替换特定标记
        system_prompt = self._generate_system_prompt(prompt_type=prompt_type, db_context=db_context,
                                                     role_status=role_status, user=user_name, char=role_name)

        # 获取消息历史
        messages = db_context.message_memory.buffer_with_langchain_msg_model(guid, count=10)
        print(messages)
        messages.append(HumanMessage(content=query))

        # 构造聊天提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="message"),
        ])

        # 创建快速回复链
        fast_chain = prompt | llm

        results = ""
        response_metadata = None

        # 异步流式处理回复
        async for r in fast_chain.astream({"message": messages}):
            results += r.content
            response_metadata = r.response_metadata
            # 封装并发送数据
            data_to_send = json.dumps({"action": "快速回复", "text": r.content}, ensure_ascii=False)
            yield data_to_send
        if remember:
            human_message = Message(user_guid=guid, type="human", role=user_name, message=query,
                                    generate_from="GameUser")
            ai_message = Message(user_guid=guid, type="ai", role=role_name, message=results, generate_from="快速回复")
            messages = [human_message, ai_message]
            await self.remember(messages=messages, db_context=db_context)

    async def rute_retriever(self, guid:str,user_name,role_name, query: str,role_status:str, db_context: DBContext,llm:BaseChatModel)->AsyncGenerator[str, None]:
        logging.info("Agent : 检索对话知识库中...")
        docs_and_scores = self.vector_db.similarity_search_with_score(query=query, k=3)
        # print(docs_and_scores)
        scores = [score for _, score in docs_and_scores]
        documents = [doc for doc, _ in docs_and_scores]
        # print("文档列表:", documents)
        combined_content = ''.join(doc.page_content for doc in documents)
        combined_content = combined_content.replace("{user}", user_name).replace("{char}", role_name)
        # print("文档内容:", combined_content)
        avg_score = sum(scores) / len(scores) if scores else 0
        print("平均相似度分数:", avg_score)
        entity = db_context.entity_memory.get_entity(guid)
        if entity is None:
            entity = Entity(entity="", summary="")
        print("实体："+entity.entity)
        opinion_memory = OpinionMemory(
            connection_string="mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent")
        # role_state = "('体力':'饥饿','精力':'疲劳','位置':'房间，沙发上','动作':'坐着')"
        history = db_context.message_memory.buffer_messages(guid,user_name,role_name, 10)
        data_to_send = json.dumps({"action": None, "text": None})
        # print("message_memory:"+history)

        if avg_score > self.similarity_threshold:
            print("Agent : 相似度分数高于阈值，使用FastChain 进行回答")
            async for r in self.response_fast(prompt_type=PromptType.FAST_CHAT, db_context=db_context,
                                              role_status=role_status,
                                              user_name=user_name, role_name=role_name, guid=guid, query=query, llm=llm):
                yield r


            # logging.info(f"Agent Fast Chain Output: {results}")
            # ai_message = Message(user_guid=guid, type="ai", role=role_name, message=results,
            #                      generate_from="快速回复",call_step=json.dumps(response_metadata))
            # db_context.message_memory.add_message(ai_message)

        else:
            print("Agent : 相似度分数低于阈值，使用DeepChain 进行回答")
            system_prompt = self._generate_system_prompt(prompt_type=PromptType.DEEP_CHAT,db_context=db_context,role_status=role_status,user=user_name,char=role_name)
            messages = db_context.message_memory.buffer_with_langchain_msg_model(guid, count=10)
            # human_message = Message(user_guid=guid, type="human", role=user_name, message="外卖呢下雨没",
            #                         generate_from="GameUser")
            # db_context.message_memory.add_message(human_message)
            # last_message = HumanMessage(content=query)
            # messages.append(last_message)
            human_message = HumanMessage(content=query)
            messages.append(human_message)
            # lm_with_tools = llm.bind(**llm_kwargs)
            lm_with_tools = llm.bind_tools([
                emotion_companion,
                fact_conversion,
                express_needs,
                information_lookup,
                opinion_evaluation,
                defensive_dialogue,
                repeat_expression
            ])
            print("message_memory:" + str(messages))
            # messages.append(HumanMessage(content=query))
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="message"),
                ]
            )
            result =""
            function_name = None
            json_buff = ""
            deep_chain = prompt | lm_with_tools
            is_use_fast_chain = False
            async for chunk in deep_chain.astream({"message": messages}):
                print(chunk)
                tool_calls = chunk.additional_kwargs.get('tool_calls')
                chunk_content = chunk.content
                # print("tool_calls:" + str(tool_calls))
                if chunk_content == "" and tool_calls:
                    # tool_calls = chunk.additional_kwargs.get('tool_calls', [])
                    for call in tool_calls:
                        function_data = call.get('function', {})
                        # print("function_data:" + str(function_data))
                        if function_data.get('name'):
                            function_name = function_data.get('name')
                            # print("function_name:" + function_name)
                        result += function_data.get('arguments', '')
                        data_to_send = json.dumps({"action": function_name, "text": None}, ensure_ascii=False)
                        yield data_to_send

                else:
                    is_use_fast_chain=True
                    logging.info(f"Agent Deep Chain 生成策略异常,使用快速回复: {chunk.content}")
                    break
            if is_use_fast_chain:
                async for r in self.response_fast(prompt_type=PromptType.FAST_CHAT, db_context=db_context,
                                                  role_status=role_status,
                                                  user_name=user_name, role_name=role_name, guid=guid, query=query,
                                                  llm=llm):
                    yield r
            else:
                # 异步生成结束后，检查result是否为有效的JSON
                try:
                    json_object = json.loads(result)
                    tool_result = ""
                    async for chunk in await self.use_tool_by_name(guid=guid,
                                                                       user_name=user_name,
                                                                       role_name=role_name,
                                                                       role_status=role_status,
                                                                       db_context=db_context,
                                                                       action_name=function_name,
                                                                       action_input=json_object
                                                                       ):
                        data_to_send = json.dumps({"action": function_name, "text": chunk}, ensure_ascii=False)
                        tool_result += chunk
                        yield data_to_send
                    human_message = Message(user_guid=guid, type="human", role=user_name, message=query,generate_from="GameUser")
                    ai_message = Message(user_guid=guid, type="ai", role=role_name, message=tool_result,
                                             generate_from=function_name,
                                             call_step=json.dumps(json_object, ensure_ascii=False))

                    db_context.message_memory.add_messages([human_message,ai_message])
                    print("Result is valid JSON.")
                except json.JSONDecodeError:
                    logging.error("Result is not valid JSON.使用fastchain")
                    async for r in self.response_fast(prompt_type=PromptType.FAST_CHAT, db_context=db_context,
                                                      role_status=role_status,
                                                      user_name=user_name, role_name=role_name, guid=guid, query=query,
                                                      llm=llm):
                        yield r



            # ai_message = Message(user_guid=guid, type="ai", role=role_name, message=result,
            #                      generate_from=function_name, call_step=)
            # db_context.message_memory.add_message(ai_message)


            #     if chunk.additional_kwargs is None:
            #         data_to_send = json.dumps({"action": None, "text": "哎呀呀~信号解析出错啦@￥##%！&"}, ensure_ascii=False)
            #         yield data_to_send
            #     if chunk.content != None and chunk.content.strip() != "":
            #         print("chunk.content:" + chunk.content)
            #         function_name = "普通回复"
            #         result += chunk.content
            #         data_to_send = json.dumps({"action": None, "text": chunk.content}, ensure_ascii=False)
            #         yield data_to_send
            #
            #     else:
            #         tool_calls = chunk.additional_kwargs.get('tool_calls', [])
            #         print("tool_calls:" + str(chunk))
            #
            #         for call in tool_calls:
            #             function_data = call.get('function', {})
            #             # print("function_data:" + str(function_data))
            #             if function_data.get('name'):
            #                 function_name = function_data.get('name')
            #                 # print("function_name:" + function_name)
            #             if function_name == "普通回复" :
            #                 # print("function_data:" + str(function_data))
            #
            #                 chunk = function_data.get('arguments')
            #                 result += chunk
            #                 data_to_send = json.dumps({"action": None, "text": chunk}, ensure_ascii=False)
            #                 yield data_to_send
            #             else:
            #                 result += function_data.get('arguments', '')
            #                 data_to_send = json.dumps({"action": function_name, "text": None}, ensure_ascii=False)
            #                 yield data_to_send
            #
            # if function_name is not None and function_name != "普通回复":
            #     try:
            #         arguments_json_array = json.dumps(result, ensure_ascii=False)
            #
            #         print("\nArguments解析为JSON成功，内容是:", arguments_json_array)
            #         result = ""
            #         try:
            #             async for chunk in await self.use_tool_by_name(guid=guid,
            #                                                            user_name=user_name,
            #                                                            role_name=role_name,
            #                                                            role_status=role_status,
            #                                                            db_context=db_context,
            #                                                            action_name=function_name,
            #                                                            action_input=arguments_json_array
            #                                                            ):
            #                 data_to_send = json.dumps({"action": function_name, "text": chunk}, ensure_ascii=False)
            #                 result += chunk
            #                 yield data_to_send
            #
            #             ai_message = Message(user_guid=guid, type="ai", role=role_name, message=result,
            #                                  generate_from=function_name, call_step=json.dumps(arguments_json_array))
            #             db_context.message_memory.add_message(ai_message)
            #         except:
            #             data_to_send = json.dumps(
            #                 {"action": function_name, "text": "抱歉呢~我好像被玩坏了，请稍后再试哦"})
            #             yield data_to_send
            #             ai_message = Message(user_guid=guid, type="ai", role=role_name,
            #                                  message="抱歉呢~我好像被玩坏了，请稍后再试哦",
            #                                  generate_from=function_name, call_step="Error")
            #             db_context.message_memory.add_message(ai_message)
            #             print("Arguments不是有效的JSON格式，请检查后重试。")
            #
            #
            #     except json.JSONDecodeError:
            #         data_to_send = json.dumps(
            #             {"action": function_name, "text": "抱歉呢~我收到的信号碎片好像出问题啦...等一等哦"})
            #         yield data_to_send
            #         ai_message = Message(user_guid=guid, type="ai", role=role_name,
            #                              message="抱歉呢~我收到的信号碎片好像出问题啦...等一等哦",
            #                              generate_from=function_name, call_step="Error")
            #         db_context.message_memory.add_message(ai_message)
            #         print("Arguments不是有效的JSON格式，请检查后重试。")
            #
            #         # logging.info(f"Agent Deep Chain Output: {strategy_output}")
            # else:
            #     # print("普通回复")
            #     # print("result:" + result)
            #     try:
            #         result = json.loads(result)
            #         ai_message = Message(user_guid=guid, type="ai", role=role_name, message=result["reply_content"],
            #                              generate_from=function_name,
            #                              )
            #         db_context.message_memory.add_message(ai_message)
            #     except json.JSONDecodeError:
            #         logging.error("普通回复内容不是有效的JSON格式，已存入数据库")
            #         ai_message = Message(user_guid=guid, type="ai", role=role_name, message=result,
            #                              generate_from=function_name, call_step="Deep/Error")
            #         db_context.message_memory.add_message(ai_message)

            # return deep_chain

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
    async def memory_entity(self,guid,user_name,role_name,message_threshold,db_context:DBContext):
        message_content, message_ids = await db_context.message_memory.check_and_buffer_messages(guid, user_name,
                                                                                                 role_name,
                                                                                                 message_threshold)
        if len(message_ids) % message_threshold == 0 and len(message_ids) != 0:
            logging.info("开始更新实体记忆")
            llm = Tongyi(model_name="qwen-turbo", temperature=0.7,dashscope_api_key="sk-dc356b8ca42c41788717c007f49e134a")
            entity = db_context.entity_memory.get_entity(guid)
            output_parser = StrOutputParser()
            if entity is None:
                entity = Entity(entity=user_name, summary="", user_guid=guid)
            info_with_entity = ENTITY_SUMMARIZATION_PROMPT.replace("{entity}", entity.entity)
            entity_with_history = info_with_entity.replace("{history}",message_content)
            entity_with_summary = entity_with_history.replace("{summary}", entity.summary)
            entity_prompt_template = PromptTemplate(template=entity_with_summary, input_variables=["input"])
            reflexion_chain = entity_prompt_template | llm | output_parser
            entity_output = ""
            async for chunk in reflexion_chain.astream({"input": ":"}):
                entity_output += chunk
                print(f"{chunk}", end="|", flush=True)
            entity.summary = entity_output
            db_context.entity_memory.add_entity(entity)
            logging.info(f"Agent 实体更新记忆: {entity}")
        else:
            logging.info("Agent 实体更新记忆: 跳过")

    async def response(self, guid:str ,user_name,role_name,input_text: str,role_status,db_context: DBContext,llm:BaseChatModel) -> AsyncGenerator[str, None]:

        # 初始化检索链
        # retriever_lambda = RunnableLambda(self.rute_retriever)
        # retriever_chain = retriever_lambda
        # human_message = Message(user_guid=guid, type="human", role=user_name, message=input_text,generate_from="GameUser")
        # logging.info(f"{guid},User Input: {input_text}")  # 记录用户输入的日志
        # db_context.message_memory.add_message(human_message)
        async for chunk in self.rute_retriever(guid=guid,user_name=user_name,role_name=role_name, query=input_text,role_status=role_status,db_context=db_context,llm=llm):
            yield chunk
        asyncio.create_task(self.memory_summary(guid, user_name, role_name, 10, db_context))
        asyncio.create_task(self.memory_entity(guid, user_name, role_name, 10, db_context))


        # step_message = ""
        # final_output = ""  # 用于存储最终输出字符串
        # self.user_input = input_text  # 存储用户输入
        #
        # human_message = Message(user_guid=guid, type="human", role=user_name, message=input_text,generate_from="GameUser")
        # logging.info(f"{guid},User Input: {input_text}")  # 记录用户输入的日志
        # db_context.message_memory.add_message(human_message)
        # # self.history.add_message_with_uid(guid=guid,message=HumanMessage(content=input_text))
        # # 通过检索链异步获取响应片段，并累加到最终输出
        # # async for chunk in retriever_chain.astream(input_text):
        # #     final_output += chunk
        # #     yield chunk
        #     # print(chunk, end="|", flush=True)
        #
        # def handle_output(output):
        #     """
        #     处理检索链的输出，尝试将其解析为JSON，失败则视为普通文本输出。
        #
        #     :param output: 检索链的输出文本。
        #     :return: 解析后的JSON对象或原始文本。
        #     """
        #     try:
        #         json_output = json.loads(output)
        #         logging.info(f"Agent Action: {json_output}")
        #         # self.history.add_ai_message(message)
        #         return json_output
        #     except json.JSONDecodeError:
        #         logging.info("Agent Action: Use FastChain")
        #         return output
        #
        # final_json_output = handle_output(final_output)  # 处理最终的检索链输出
        #
        # if isinstance(final_json_output, dict):
        #     strategy_output = ""
        #     # 如果输出是字典，则进一步通过深度处理链处理，并累加响应
        #     async for chunk in await self.route_post_deep_chain(guid=guid,user_name=user_name,role_name=role_name, input=final_json_output,role_status=role_status,db_context=db_context):
        #         strategy_output += chunk
        #         # print(f"{chunk}", end="|", flush=True)
        #         yield chunk
        #     # logging.info(f"Agent Deep Chain Output: {strategy_output}")
        #
        #     # 将字典转换为JSON格式的字符串
        #     # input_str = json.dumps(json_output["input"], ensure_ascii=False)
        #     concatenated_values = ''
        #     for key, value in final_json_output["input"].items():
        #         concatenated_values += f"{key}={value}" + ','
        #     step_message = f"Action: {final_json_output['action']} - Input: {concatenated_values}"
        #
        #     ai_message = Message(user_guid=guid, type="ai", role=role_name, message=strategy_output,generate_from="Chat-DeepChain",call_step=step_message)
        #     db_context.message_memory.add_message(ai_message)
        #     logging.info(f"Agent Deep Chain Output: {strategy_output}")
        #     # self.history.add_message_with_uid(guid=guid,message=AIMessage(content=strategy_output,generate_from="DeepChain" , call_step=step_message))
        # else:
        #     # 如果输出不是字典，则视为快速链输出
        #     logging.info(f"Agent Fast Chain Output: {final_output}")
        #     ai_message = Message(user_guid=guid, type="ai", role=role_name, message=final_output,
        #                          generate_from="Chat-FastChain/Deep",call_step=step_message)
        #     db_context.message_memory.add_message(ai_message)
        #
        # # entity_memory = EntityMemory(
        # #     connection_string="mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent")
        #
        # entity = db_context.entity_memory.get_entity(guid)
        # output_parser = StrOutputParser()
        # if entity is None:
        #     entity = Entity(entity=user_name,summary="",user_guid=guid)
        # info_with_entity = ENTITY_SUMMARIZATION_PROMPT.replace("{entity}",entity.entity)
        # entity_with_history = info_with_entity.replace("{history}",db_context.message_memory.buffer_messages(guid,user_name,role_name,count=10))
        # entity_with_summary = entity_with_history.replace("{summary}",entity.summary)
        # entity_prompt_template = PromptTemplate(template=entity_with_summary, input_variables=["input"],)
        # reflexion_chain = entity_prompt_template | self.fast_llm | output_parser
        # entity_output=""
        # async for chunk in reflexion_chain.astream({"input":""}):
        #     entity_output += chunk
        #     print(f"{chunk}", end="|", flush=True)
        # entity.summary = entity_output
        # db_context.entity_memory.add_entity(entity)
        # logging.info(f"Agent 实体更新: {entity}")
        #
    async def write_diary(self,user_name,role_name,guid:str,date_start,date_end,llm:BaseLLM,db_context:DBContext) -> AsyncGenerator[str, None]:
        system_prompt = self._generate_system_prompt(prompt_type=PromptType.WRITE_DIARY,db_context=db_context,guid=guid,role=self.base_info,user=user_name,char=role_name,date_start=date_start,date_end=date_end)
        prompt_template = PromptTemplate.from_template(system_prompt)
        output_parser = StrOutputParser()
        final_diary = ""
        diary_chain =  prompt_template | llm | output_parser
        async for chunk in diary_chain.astream({}):
            final_diary += chunk
            yield chunk
        logging.info(f"Agent Write Diary: {final_diary[:30]}...")
        db_context.message_memory.add_message(Message(user_guid=guid, type="event", role="系统事件", message=f"{role_name}写了一篇日记：{final_diary}",
                                                        generate_from="WriteDiary"))


    def _generate_system_prompt(self,prompt_type: PromptType,db_context:DBContext,guid:str=None,role=None,role_status=None, user=None,char=None,date_start=None,date_end=None):
        if prompt_type == PromptType.EVENT:
            recent_event =db_context.message_summary.buffer_summaries(guid,20)
            system_prompt = EVENT_PROMPT.format(
                role=role,
                recent_event=recent_event,
                user=user,
                char=char
            )
            return system_prompt
        if prompt_type == PromptType.WRITE_DIARY:
            history_buffer = db_context.message_memory.buffer_messages(guid,user_name=user,role_name=char,count=50,start_date=date_start,end_date=date_end)
            system_prompt = WRITE_DIARY_PROMPT.format(
                role=role,
                user=user,
                char=char,
                history=history_buffer

            )
            return system_prompt
        if prompt_type == PromptType.FAST_CHAT:
            # Setup chains
            opinion_memory = OpinionMemory(
                connection_string="mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent")
            system_prompt = self.character_info.format(
                role_state=role_status,
                user=user,
                char=char,
                memory_of_user=db_context.entity_memory.get_entity(guid),
                environment="",
                recent_event=db_context.message_summary.buffer_summaries(guid, 20),
                opinion=opinion_memory.buffer(guid, 10),
            )
            return system_prompt
        if prompt_type == PromptType.DEEP_CHAT:
            # Setup chains
            opinion_memory = OpinionMemory(
                connection_string="mysql+pymysql://db_role_agent:qq72122219@182.254.242.30:3306/db_role_agent")
            system_prompt = self.tuji_info.format(
                role_state=role_status,
                user=user,
                char=char,
                memory_of_user=db_context.entity_memory.get_entity(guid),
                environment="",
                recent_event=db_context.message_summary.buffer_summaries(guid, 20),
                opinion=opinion_memory.buffer(guid, 10),
            )
            return system_prompt


    async def event_response(self,user_name,role_name,llm:BaseChatModel,guid:str,event: str,db_context:DBContext) -> AsyncGenerator[str, None]:

        system_message = Message(user_guid=guid, type="event", role="系统事件", message=event,
                                 generate_from="game")
        db_context.message_memory.add_message(system_message)
        messages = db_context.message_memory.buffer_with_langchain_msg_model(guid,count=10)
        # messages.append(HumanMessage(content=event))
        print(messages)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",self._generate_system_prompt(PromptType.EVENT,db_context,guid,role_name,user_name,role_name)),
                MessagesPlaceholder(variable_name="message"),
            ]
        )
        chain = prompt | llm
        results=""
        response_metadata =None
        async for r in chain.astream({"message": messages}):
            results += r.content
            response_metadata=r.response_metadata
            yield r.content

        response_metadata=json.dumps(response_metadata)


        ai_message = Message(user_guid=guid, type="ai", role=role_name, message=results,
                             generate_from="", call_step=response_metadata)
        db_context.message_memory.add_message(ai_message)
        logging.info(f"Agent System Event: {event}")
        logging.info(f"Agent System Event Response: {results}")
        asyncio.create_task(self.memory_summary(guid, user_name, role_name, 10, db_context))
        asyncio.create_task(self.memory_entity(guid, user_name, role_name, 10, db_context))





        #
        # info_with_role = EVENT_PROMPT.replace("{role}",self.base_info)
        # event_recent = info_with_role.replace("{recent_event}",db_context.message_summary.buffer_summaries(guid,20))
        # info_with_history = event_recent.replace("{history}",db_context.message_memory.buffer_messages(guid,user_name,role_name,count=10))
        # info_name = info_with_history.replace("{user}", user_name).replace("{char}", role_name)
        # # print(info_name)
        # prompt_template = PromptTemplate(template=info_name, input_variables=["event"])
        # output_parser = StrOutputParser()
        # event_chain = prompt_template | llm | output_parser
        # results =""
        #
        # async for chunk in event_chain.astream({"event":event}):
        #     results+=chunk
        #     yield chunk
        #
        # system_message = Message(user_guid=guid, type="system", role="系统事件", message=event,
        #                      generate_from="SystemEvent")
        #
        # ai_message = Message(user_guid=guid, type="ai", role=role_name, message=results,
        #                      generate_from="SystemEvent")
        # db_context.message_memory.add_messages([system_message,ai_message])
        # logging.info(f"Agent System Event: {event}")
        # logging.info(f"Agent System Event Response: {results}")


    async def balderdash(self, user_name,role_name,role_info,guid:str,exception,user_input,db_context:DBContext,llm=None):
        logging.info(f"Agent Balderdash: 触发异常，胡言乱语中...")
        if llm is None:
            llm = Tongyi(model_name="qwen-turbo", temperature=0.7,
                         dashscope_api_key="sk-dc356b8ca42c41788717c007f49e134a")
            prompt_template = PromptTemplate(template=BALDERDASH_TEMPLATE, input_variables=["role_info","input","exception"])
            output_parser = StrOutputParser()
            chain = prompt_template |llm | output_parser
            results="胡言乱语中..."
            action_name = "胡言乱语"
            async for chunk in chain.astream({"role_info":role_info,"input":user_input,"exception":exception}):
                results+=chunk
                yield chunk


            ai_message = Message(user_guid=guid, type="ai", role=role_name, message=results,
                                 generate_from=action_name)
            db_context.message_memory.add_message(ai_message)




    async def memory_summary(self,user_name,role_name,guid:str,message_threshold:int,db_context:DBContext,llm=None):
        print(f"Agent Summary: 判断是否需要生成摘要...")
        message_content, message_ids =await db_context.message_memory.check_and_buffer_messages(guid, user_name, role_name,
                                                                                           message_threshold)
        if len(message_ids) % message_threshold == 0 and len(message_ids) != 0:
            print("生成摘要...")
            if llm is None:
                llm = Tongyi(model_name="qwen-turbo", temperature=0.7,
                             dashscope_api_key="sk-dc356b8ca42c41788717c007f49e134a")
            prompt_template = PromptTemplate(template=EVENT_SUMMARY_TEMPLATE, input_variables=["history"])
            output_parser = StrOutputParser()

            chain = prompt_template | llm | output_parser
            results = ""
            async for chunk in chain.astream({"history": message_content}):
                results += chunk

            # 保存摘要到Message_Summary表，并关联消息ID
            summary = Message_Summary(user_guid=guid,summary=results)
            db_context.message_summary.add_summary(message_summary=summary,message_ids=message_ids)
            logging.info(f"Agent Summary: {summary}")
            # print(message_summary_id)
            # db_context.message_memory.bind_summary_id_to_messages(message_ids, message_summary_id)
        else:
            print("不需要生成摘要")



    def perform_task(self, task: str, data: dict) -> int:
        return 200

    async def remember(self, messages: List[BaseMessage],db_context:DBContext):
        db_context.message_memory.add_messages(reversed(messages))


    def recall(self, key: str) -> any:
        pass


