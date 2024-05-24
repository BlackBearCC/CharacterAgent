FROM python:3.9

# 设置工作目录为 /app
WORKDIR /app

# 将当前目录下的文件复制到容器的 /app 目录
COPY . /app

# 安装 requirements.txt 中列出的依赖
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir uvicorn[standard] fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装uvicorn的标准功能集，确保了websocket
# 预下载 NLTK 数据
RUN pip install nltk
RUN python -m nltk.downloader punkt

# 让容器监听80端口
EXPOSE 80



# 运行应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]