import logging

from langchain_core.tools import tool


@tool("情感陪伴")
def emotion_companion(emotion_status: str, emotion_intensity: str, user_needs: str, critical_context: str,
                      reply_instruction: str) -> str:
    """
    当你判断用户需要情感陪伴时非常有用。

    Args:
        emotion_status (str): 用户情感状态
        emotion_intensity (str): 用户情感强度
        user_needs (str): 分析的用户现在真实需求
        critical_context (str): 可参考的关键上下文信息
        reply_instruction (str): 结合人设和上下文和以上的值，输出你的指导回复的关键词组（不超过10个字）
    """
    # 实现情感陪伴工具的逻辑
    return "情感陪伴回复"

@tool("事实转换")
def fact_conversion(real_world_entity: str, character_perspective: str, reply_instruction: str) -> str:
    """
    当你需要将现实世界的信息（著名人物、地点、事件、物品等）转化为角色视角下的对应物时非常有用。

    Args:
        real_world_entity (str): 需要转换的现实世界实体（人物、地点、事件、物品）
        character_perspective (str): 你的角色视角或背景中的所能对应的实体
        reply_instruction (str): 结合人设和上下文和以上的值，输出你的指导回复的关键词组（不超过10个字）
    """
    # 实现事实转换工具的逻辑
    return "事实转换回复"

@tool("表达诉求")
def express_needs(current_role_state: str, maslow_hierarchy_level: str, reply_instruction: str) -> str:
    """
    当你需要基于自身状态和马斯洛需求理论的需求层级表达自身诉求时非常有用。

    Args:
        current_role_state (str): 角色自身当前的状态
        maslow_hierarchy_level (str): 需求层级，根据马斯洛需求层次理论
        reply_instruction (str): 结合人设和上下文和以上的值，输出你的指导回复的关键词组（不超过10个字）
    """
    # 实现表达诉求工具的逻辑
    return "表达诉求回复"
@tool("信息查找")
def information_lookup(storage_type: str, reply_instruction: str) -> str:
    """
    只能用于查找和回答有关冰箱或储物柜内物品的具体信息，如数量、内容、位置等。

    Args:
        storage_type (str): 储存类型（冰箱或储物柜）
        reply_instruction (str): 结合人设和上下文，输出你的指导回复的关键词组（不超过10个字，不可以编造查询结果！）
    """
    # 实现信息查找工具的逻辑
    return "信息查找回复"


@tool("观点评价")
def opinion_evaluation(opinion_id: int, entity_or_opinion: str, evaluation_scale: str, evaluation_reason: str,
                       reply_instruction: str) -> str:
    """
    对特定观点或实体发表评价，并用1-5的Likert量表进行评分。

    Args:
        opinion_id (int): 引用观点的ID
        entity_or_opinion (str): 被评价的实体或观点
        evaluation_scale (str): 评价的Likert量表范围（1-5）
        evaluation_reason (str): 评价的原因
        reply_instruction (str): 结合人设和上下文和以上的值，输出你的指导回复的关键词组（不超过10个字）
    """
    # 实现观点评价工具的逻辑
    return "观点评价回复"

@tool("防御对话")
def defensive_dialogue(attack_type: str, core_attributes: str, reply_instruction: str) -> str:
    """
    当面对攻击或诱导时保护对话的策略，保持角色的核心属性和知识范围。

    Args:
        attack_type (str): 攻击或诱导对话的类型及内容
        core_attributes (str): 角色的核心属性
        reply_instruction (str): 结合人设和上下文和以上的值，输出你的指导回复的关键词组（不超过10个字）
    """
    # 实现防御对话工具的逻辑
    return "防御对话回复"
@tool("重复表达")
def repeat_expression(history_question: str, key_role_state: str, expression_attitude: str,
                      reply_instruction: str) -> str:
    """
    当你发现用户重复提问时非常有用，对历史中已经回答过且重复的问题进行回应，展示角色的情绪和态度。

    Args:
        history_question (str): 历史中已回答过的问题
        key_role_state (str): 角色自身的关键状态
        expression_attitude (str): 综合考虑角色对重复问题的情绪和态度
        reply_instruction (str): 结合人设和上下文和以上的值，输出你的指导回复的关键词组（不超过10个字）
    """
    # 实现重复表达工具的逻辑
    return "重复表达回复"