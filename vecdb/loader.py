import os
from langchain_community.document_loaders import PyMuPDFLoader

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

# 设置路径
data_path = 'data'
vec_path =  data_path+'/vector_db/chroma'

# 初始化向量化模型
# model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

# 初始化Chroma数据库
# vector_store = Chroma(persist_directory=data_path,embedding_function=OpenAIEmbeddings())

# 加载Excel文件
def process_excel(file_path):
    loader = UnstructuredExcelLoader(file_path)
    data = loader.load()
    return data

# 加载PDF文件
def process_pdf(file_path):
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    return data

# 处理并向量化文件
def process_file(file_path):
    print(file_path)
    if file_path.endswith('.xlsx'):
        text = process_excel(file_path)
    elif file_path.endswith('.pdf'):
        text = process_pdf(file_path)
    else:
        raise ValueError("Unsupported file type")
    return text

if __name__ == "__main__":
    # 遍历data文件夹中的所有文件
    # texts = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.xlsx') or file.endswith('.pdf'):
                text = process_file(file_path)
                docs = text_splitter.split_documents(text)
                db = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory=vec_path)