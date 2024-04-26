from typing import Any, Dict, List, AsyncGenerator

from app.core.abstract_Agent import AbstractAgent


class CharacterAgent(AbstractAgent):

    def __init__(self, name: str, model: Any):
        self.name = name
        self.model = model
        self.memory = {}

    async def _async_response_generator(self, input_text: str) -> AsyncGenerator[str, None]:
        """
        返回一个异步生成器，该生成器基于输入文本异步地产生模型的响应。

        参数:
        input_text: str - 传递给模型以生成响应的输入文本。

        返回值:
        AsyncGenerator - 该生成器将异步地从模型中获取响应。
        """

        try:
            async for event_data in self.model.async_sync_call_streaming(input_text):
                yield event_data
        except Exception as e:
            # 更复杂的错误处理逻辑，例如错误日志记录
            print(f"生成响应时发生错误: {e}")
            # 根据实际情况，可以选择抛出自定义异常或进行其他错误处理
            raise

    def response(self, input_text: str) -> AsyncGenerator[str, None]:
        """
        提供一个快速返回接口，用于直接异步地从模型获取响应。

        参数:
        input_text: str - 传递给模型以生成响应的输入文本。

        返回值:
        AsyncGenerator - 通过内部调用异步生成器来实现，简化外部调用的复杂性。
        """


        # 调用异步生成器方法，并直接返回生成器
        # 外部调用此方法时，需要在异步上下文中使用，例如 async with 或 async for
        return self._async_response_generator(input_text)
        # return self.model.normal_call(input_text)

    async def response_stream(self,input_text: str):
        async for chunk in self.model.astream_with_langchain(input_text):
            print(chunk, end="|", flush=True)

    async def response_stream_with_retriever(self,input_text: str, retriever):
        async for chunk in self.model.astream_with_langchain_RAG(retriever,input_text):
            print(chunk, end="|", flush=True)
    def perform_task(self, task: str, data: dict) -> int:
        return 200

    def remember(self, key: str):
        return 200

    def recall(self, key: str) -> any:
        pass


