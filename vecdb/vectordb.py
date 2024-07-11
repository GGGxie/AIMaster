from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .loader import vec_path

# vectordb = Chroma(embedding_function=ZhipuAIEmbeddings(), persist_directory=persist_directory)
vectordb = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory=vec_path)

# 加载向量数据库
def get_vectordb():
    return vectordb