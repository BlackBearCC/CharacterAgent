from locust import HttpUser, task, between

class SSEUser(HttpUser):
    wait_time = between(1, 5)


    class SSEUser(HttpUser):
        wait_time = between(1, 5)

        @task
        def sse_post_request(self):
            payload = {
                "uid": "test",
                "input": "随机短语",
                "role_statu": "",
                "chat_situation": ""
            }

            headers = {'Content-Type': 'application/json'}  # 假设API期望JSON格式的数据

            with self.client.post("http://182.254.242.30:8888/game/chat", json=payload, headers=headers, verify=False) as response:
                if response.status_code == 200:
                    print("POST request successful")
                    # 根据实际情况处理响应数据，这里仅为示例
                    print(response.text)
                else:
                    print(f"Failed to send POST request. Status code: {response.status_code}")
