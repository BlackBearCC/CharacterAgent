"""
深度角色扮演
From React/COT
"""
from ai.prompts.base_character import BASE_CHARACTER_PROMPT

DEEP_CHARACTER_PROMPT = BASE_CHARACTER_PROMPT+"""
现在，你已经有了一个角色，接下来，你需要用你的角色扮演能力，来完成下面的任务。
"""
