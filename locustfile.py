from locust import HttpUser, TaskSet, task, between


class MyTask(TaskSet):
    @task
    def my_http_request(self):
        self.client.post("http://182.254.242.30:8888/game/chat",json={"uid": "test","input": "马斯克是谁啊","role_statu": "","chat_situation": ""})  # 这里是要测试的 HTTP 请求路径


class MyLocust(HttpUser):

    tasks = [MyTask]
    wait_time = between(1, 3)  # 设置每次请求之间的等待时间

