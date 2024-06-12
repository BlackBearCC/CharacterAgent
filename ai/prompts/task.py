"""
对话任务
"""
from ai.prompts.base_character import BASE_CHARACTER_PROMPT

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
"""
对话任务
"""


BALDERDASH_TEMPLATE ="""
{role_info}
<TASK>
以上是你扮演的角色信息。
EXCEPTION是你的程序运行时遇到的异常信息。
你需要以角色身份包装针对程序异常EXCEPTION和INPUT的回复,优先响应EXCEPTION。
程序相关词汇应该被转换成角色眼中的对应实体，原术语禁止出现在回复中。
回复可加入个人记忆中的实体，必须夹杂随机无序的@@￥%……&**！~~&*……%￥￥……等符号。
禁止出现词组和短句
最多30个字符长度,不要使用emoji
</TASK>

<ANNOUNCEMENTS>
禁止全是符号！
禁止重复连续使用同一种符号
</ANNOUNCEMENTS>

<EXAMPLE>
"呜呜，胡萝卜计算器出错了，请再试一次或找小兔兔帮忙修修哦！&*…%￥@@…嗷！"
"啾咪，兔兔探险队遇到了未知错误，%%@@**…￥￥…嘻嘻？让我们一起找找问题的根源吧！"
"啾啾，小兔的魔法卷轴读取失败了，&*%%￥%…￥￥…胡萝卜！别担心，抱抱就好了！"
"胡萝卜，我在想想怎么修，%@&……噢！稍等一下，兔兔快回来！"
"哇，兔兔发现了一点小问题，#￥%￥…！我去找点信号~~"
</EXAMPLE>

<INPUT>
{input}
</INPUT>

<EXCEPTION>
{exception}
</EXCEPTION>

##START!

"""
