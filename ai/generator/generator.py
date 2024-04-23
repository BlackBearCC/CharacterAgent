import os
import json
import aiohttp

class Generator:
    def __init__(self, api_key=None):
        # self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", None)
        # if not self.api_key:
        #     raise ValueError("DASHSCOPE_API_KEY environment variable is not set.")
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    async def async_sync_call_streaming(self, model, prompt_text, callback=None, session_id=None, query=None):
        headers = {
            # 'Authorization': f'Bearer {self.api_key}',
            'Authorization': f'Bearer sk-dc356b8ca42c41788717c007f49e134a',
            'Content-Type': 'application/json',
            'X-DashScope-SSE': 'enable',
        }
        data = model.prepare_data(prompt_text)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.api_url, headers=headers, json=data) as response:
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
                print(f"HTTP请求错误: {e}")
