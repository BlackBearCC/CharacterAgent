from abc import ABC, abstractmethod

from langchain.memory import ConversationBufferMemory


class AbstractAgent(ABC):
    """
    抽象代理类，定义了一个代理的基本行为。
    """

    @abstractmethod
    def response(self,uid:str, input_text: str) -> str:
        """
        处理来自用户的输入，并生成相应的响应。

        参数:
        input_text (str): 用户的输入文本。

        返回:
        str: 生成的响应文本。
        """
        pass

    @abstractmethod
    def perform_task(self, task: str, data: dict) -> int:
        """
        执行指定的任务。

        参数:
        task (str): 任务的名称或标识。
        data (dict): 执行任务所需的数据或信息。
        """
        pass

    @abstractmethod
    def remember(self, key: str ):
        """
        保存一条信息，以供将来回忆。

        参数:
        key (str): 信息的键，用于回忆信息时使用。
        """
        pass

    @abstractmethod
    def recall(self, key: str) -> any:
        """
        根据键回忆之前保存的信息。

        参数:
        key (str): 需要回忆的信息的键。

        返回:
        any: 与键对应的信息，如果找不到，则返回None。
        """
        pass
