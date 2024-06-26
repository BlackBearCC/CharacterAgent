"""
反思
From React/COT
"""


REFLEXION = """

"""

ENTITY_SUMMARIZATION_PROMPT = """
<TASK>
你是一个AI助手，帮助人类记录生活中相关人物、地点和概念的事实。根据对话历史，更新"实体"部分中提供的实体的摘要。如果您是首次编写摘要，返回一个句子。
</TASK>

<WORKFLOW>
更新应仅包括关于提供的实体在对话历史中被提到的事实，并且应仅包含关于该实体的事实。
如果没有关于给定实体的新信息，或者信息不值得长期记住，那么应保持现有摘要不变。
如果有要更新的内容，请在保留原实体内容的基础上添加新内容，严格禁止直接覆盖实体。
</WORKFLOW>

<ANNOUNCEMENTS>
禁止输出WORKFLOW之外的其他内容，禁止输出你的推理步骤，只返回结论内容。
</ANNOUNCEMENTS>
<INFO>
完整对话历史（供上下文参考）： {history}
需要总结的实体： {entity}
{entity}的现有总结： {summary}
</INFO>

##START!
{input}
更新后的总结：
"""