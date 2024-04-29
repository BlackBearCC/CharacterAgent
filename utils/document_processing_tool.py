import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter


class DocumentProcessingTool:
    """
    文档处理工具类，用于加载和处理文本文件。

    参数:
    - file_path: 文档或文件夹的路径。
    - model_name: 用于文本处理的模型名称，默认为"thenlper/gte-small-zh"。
    - chunk_size: 每个处理块的大小，默认为80。
    - chunk_overlap: 处理块之间的重叠大小，默认为6。
    - normalize_embeddings: 是否标准化嵌入，默认为True。
    - device: 模型运行的设备，默认为'cpu'。
    - glob_pattern: 文件路径匹配模式，默认为"**/*.txt"，即匹配所有文本文件。
    - show_progress: 是否显示处理进度，默认为True。
    - use_multithreading: 是否使用多线程处理，默认为True。
    """
    def __init__(self, file_path, model_name="thenlper/gte-small-zh", chunk_size=80, chunk_overlap=6,
                 normalize_embeddings=True, device='cpu', glob_pattern="**/*.txt", show_progress=True,
                 use_multithreading=True):
        # 初始化参数
        self.file_path = file_path
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.glob_pattern = glob_pattern
        self.show_progress = show_progress
        self.use_multithreading = use_multithreading
        # 初始化 vectordb 为 None，用于存储处理结果
        self.vectordb = None

    def process_and_build_vector_db(self):
        """
        处理文件，分割文本，创建嵌入，并构建向量数据库。

        首先检查文件路径是否存在，然后加载文档，将文档分割成片段，创建嵌入模型，
        最后使用分割的文档和嵌入模型构建向量数据库。

        返回:
            Retriever: 构建好的向量数据库检索器实例。
        """
        # 检查文件路径是否存在
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The provided path {self.file_path} does not exist.")

        # 加载文档
        loader = DirectoryLoader(self.file_path, glob=self.glob_pattern, show_progress=self.show_progress,
                                 use_multithreading=self.use_multithreading)
        documents = loader.load()

        # 分割文档为片段
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(documents)

        # 创建嵌入模型
        embedding_model = HuggingFaceEmbeddings(model_name=self.model_name, model_kwargs={'device': self.device},
                                                encode_kwargs={'normalize_embeddings': self.normalize_embeddings})

        # 构建向量数据库
        self.vectordb = Chroma.from_documents(documents=docs, embedding=embedding_model)

        return self.vectordb.as_retriever()

