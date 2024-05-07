from langchain.tools import BaseTool
from typing import Callable
import random


class DialogueTool(BaseTool):
    """基类,实现对话策略的调用逻辑"""

    def _run(self, query: str, run_manager=None) -> str:
        """使用合适的策略处理查询,返回响应"""
        strategy_func = self.strategy(query)
        if strategy_func is None:
            return "无法确定适当的对话策略。"
        else:
            return strategy_func()

    def strategy(self, query: str) -> Callable:
        """根据查询内容选择合适的策略"""
        raise NotImplementedError


class EmotionCompanionTool(DialogueTool):
    """情感陪伴策略"""
    name = "emotion_companion"
    description = "根据用户的情绪调整语气和内容"
    def strategy(self, query: str) -> Callable:
        # 使用 NLP 模型检测查询情感,选择相应的响应策略

        ...


class FactTransformTool(DialogueTool):
    """事实转换策略"""
    name = "fact_transform"
    description = "将事实转换为角色视角"


    def strategy(self, query: str) -> Callable:
        # 解析查询,将事实转换为角色视角
        ...


class ExpressionTool(DialogueTool):
    """表达诉求策略"""

    def strategy(self, query: str) -> Callable:
        # 根据角色设定,表达合适的需求
        print("调用表达诉求策略")
        ...


class InformationTool(DialogueTool):
    """信息查找策略"""

    def strategy(self, query: str) -> Callable:
        # 在知识库中查找答案
        print("调用信息查找策略")
        ...


class OpinionTool(DialogueTool):
    """观点评价策略"""

    def strategy(self, query: str) -> Callable:
        # 使用 NLP 模型对查询进行情感分析
        # 给出 1-5 分并解释原因
        print("调用观点评价策略")
        ...


class DefenseTool(DialogueTool):
    """防御对话策略"""

    def strategy(self, query: str) -> Callable:
        # 检测是否有违背角色设定的内容
        # 如果有,给出拒绝响应
        print("调用防御对话策略")
        ...


class RepeatTool(DialogueTool):
    """重复表达策略"""

    def strategy(self, query: str) -> Callable:
        # 检测是否重复之前的查询
        # 如果是,调侃并表达情绪
        print("调用重复表达策略")
        ...


class TopicTool(DialogueTool):
    """话题激发策略"""

    topics = [...]  # 话题池

    def strategy(self, query: str) -> Callable:
        # 如果查询无聊,从话题池随机选一个新话题
        print("调用话题激发策略")
        ...



