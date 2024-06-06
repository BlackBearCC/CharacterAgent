"""
对话任务
"""


EVENT_SUMMARY_TEMPLATE = """

<TASK>
你是一个有帮助的助手。使用提供的HISTORY来总结内容。简洁地总结对话,不要超过4句话。
</TASK>

<ANNOUNCEMENTS>
保持事件中人物状态，关系，位置，事件变化和对话核心的详细描述，准确总结事件，确保清晰易读，言简意赅。
</ANNOUNCEMENTS>

<HISTORY>
{history}
</HISTORY>

##START!
总结：
"""

