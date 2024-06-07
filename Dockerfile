FROM python:3.9

# 设置工作目录为 /app
WORKDIR /app

# 将当前目录下的文件复制到容器的 /app 目录
COPY . /app
# 复制nltk_data到容器的nltk默认数据目录
COPY nltk_data /usr/share/nltk_data
COPY .env /app/.env


# 安装必要的库
#RUN pip install nltk -i https://pypi.tuna.tsinghua.edu.cn/simple


# 安装 requirements.txt 中列出的依赖
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir uvicorn[standard] fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple
#RUN #pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

# 让容器监听80端口https://devops.aliyun.com/lingma/login?port=37510&state=2-2375a61148324a3e97bf8f79693af1fa


EXPOSE 80


#pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  && \
#pip install --no-cache-dir uvicorn[standard] fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple && \
#pip install --no-cache-dir sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple && \
CMD uvicorn main:app --host 0.0.0.0 --port 80 --workers 4
