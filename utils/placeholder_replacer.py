import re
from typing import List, Any

from langchain_core.tools import BaseTool




class PlaceholderReplacer:
    """
    工具类用于替换字符串中的占位符为指定字典中的值。
    """

    @staticmethod
    def replace_dict_placeholders(prompt_string: str, config_dict: dict) -> str:
        """
        静态方法，将字符串中的占位符{{key}}替换为配置字典中对应的值。
        如果key不存在，则保持原样返回{{key}}。

        参数:
        - prompt_string: 原始字符串，可能包含占位符。
        - config_dict: 包含键值对的字典，用于替换占位符。

        返回:
        - 替换后的字符串。
        """
        def replace(match):
            # 提取匹配的占位符key，并尝试从配置字典中获取其值
            key = match.group(1)
            # 注意：这里直接返回config_dict.get(key, "{{{key}}}")即可，无需再进行.format操作
            return config_dict.get(key, "{{{{{key}}}}}")

        # 定义正则表达式，匹配{{key}}形式的占位符
        pattern = re.compile(r"\{\{(.+?)\}\}")
        # 使用正则表达式和替换函数，替换所有占位符
        return pattern.sub(replace, prompt_string)

    @staticmethod
    def replace_tools_with_details(prompt_string: str, tools: List[Any]) -> str:
        """
        将字符串中的{tools}占位符替换为工具类的名称、描述和参数（以JSON格式）。

        参数:
        - prompt_string: 原始字符串，可能包含{tools}占位符。
        - tools: 工具类实例列表，每个实例需有name、description和params属性，params应为一个JSON字典。

        返回:
        - 替换后的字符串。
        """
        if "{tools}" in prompt_string:
            from app.core.tools.dialogue_tool import DialogueTool

            # 确保 DialogueTool 类已经被导入
            assert DialogueTool
            tools_details = "\n".join([
                f"Name: {tool.name}, "
                f"Description: {tool.description}, "
                f"Params: {{{tool.params}}}"  # 使用双大括号包裹params
                for tool in tools
            ])
            return prompt_string.replace("{tools}", tools_details)
        else:
            return prompt_string
