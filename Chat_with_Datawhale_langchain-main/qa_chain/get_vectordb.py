import sys 
# sys.path.append("../embedding") 
# sys.path.append("../database") 

from langchain_community.embeddings.openai import OpenAIEmbeddings    # 调用 OpenAI 的 Embeddings 模型
import os
from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from database.create_db import create_db,load_knowledge_db
from embedding.call_embedding import get_embedding

def get_vectordb(file_path:str=None, persist_path:str=None, embedding = "openai",embedding_key:str=None):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if persist_path is None:
        persist_path = os.path.join(BASE_DIR, "exam_vector_db", "chroma")
    if file_path is None:
        file_path = os.path.join(BASE_DIR, "exam_knowledge_db")
    embedding = get_embedding(embedding=embedding, embedding_key=embedding_key)
    if os.path.exists(persist_path):  #持久化目录存在
        contents = os.listdir(persist_path)
        if len(contents) == 0:  #但是下面为空
            #print("目录为空")
            vectordb = create_db(file_path, persist_path, embedding)
            #presit_knowledge_db(vectordb)
            vectordb = load_knowledge_db(persist_path, embedding)
        else:
            #print("目录不为空")
            vectordb = load_knowledge_db(persist_path, embedding)
    else: #目录不存在，从头开始创建向量数据库
        vectordb = create_db(file_path, persist_path, embedding)
        #presit_knowledge_db(vectordb)
        vectordb = load_knowledge_db(persist_path, embedding)

    return vectordb