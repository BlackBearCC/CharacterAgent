import os
import json
from http import HTTPStatus

import aiohttp
import dashscope

dashscope.api_key = "sk-dc356b8ca42c41788717c007f49e134a"
class QianwenModel:

    def __init__(self):
        self.system_prompt = ""


    def normal_call(self, prompt_text: str, callback=None):
        messages = [{'role': 'system', 'content':f"{self.system_prompt}" },
                    {"role": "user", "content": prompt_text}]
        response =dashscope.Generation.call(
            dashscope.Generation.Models.qwen_max,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
        )

        if response.status_code == HTTPStatus.OK:
            print(response)
            try:
                self._response_text = response['output']['choices'][0]['message']['content']
                self._usage = response['usage']
                if callback:
                    callback(self._response_text, self._usage)
            except KeyError as e:
                raise Exception(f"解析响应时出错: {e}")
        else:
            raise Exception(f"API 请求失败，状态码: {response.status_code}")


    async def async_sync_call_streaming(self, prompt_text, callback=None, session_id=None, query=None):
        # # 从环境变量读取API密钥
        # DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
        # if not DASHSCOPE_API_KEY:
        #     raise ValueError("DASHSCOPE_API_KEY environment variable is not set.")

        DASHSCOPE_API_URL = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
        headers = {
            # 'Authorization': f'Bearer {DASHSCOPE_API_KEY}',
            'Authorization': f'Bearer sk-dc356b8ca42c41788717c007f49e134a',
            'Content-Type': 'application/json',
            'X-DashScope-SSE': 'enable',
        }
        data = {
            "model": "qwen-plus",
            "temperature": 0.2,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": f"{self.system_prompt}"
                    },
                    {
                        "role": "user",
                        "content": f"{prompt_text}"
                    }
                ]
            },
            "parameters": {
            }
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(DASHSCOPE_API_URL, headers=headers, json=data) as response:
                    http_status_ok = False
                    output_text = ""
                    output_request_id = ""
                    output_content = ""
                    error_message = ""

                    if response.status == 200:
                        async for line in response.content:
                            text = line.decode('utf-8').strip()
                            if text.startswith('event: ping'):
                                continue
                            if "HTTP_STATUS/200" in text:
                                http_status_ok = True
                                continue

                            if http_status_ok and text.startswith('data:'):
                                try:
                                    json_data = json.loads(text[len('data:'):].strip())
                                    if 'output' in json_data and 'text' in json_data['output']:
                                        output_text = json_data['output']['text']
                                        output_request_id = json_data.get("request_id", "未知request_id")

                                        combined_data = {
                                            "text": output_text,
                                            "request_id": output_request_id
                                        }

                                        event_data = {"event": response.status, "data": combined_data}
                                        yield event_data
                                except json.JSONDecodeError as e:
                                    # 使用日志记录代替print
                                    print(f"SSE文本转JSON错误: {e}")

                    else:
                        event_data = {
                            "event": response.status,
                            "data": "请求失败"
                        }
                        yield event_data

                if callback and http_status_ok:
                    await callback(output_text, session_id, query)

            except aiohttp.ClientError as e:
                # 异常处理增强：捕获并处理aiohttp相关的客户端错误
                print(f"HTTP请求错误: {e}")

