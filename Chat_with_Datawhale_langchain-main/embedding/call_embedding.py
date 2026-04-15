import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from llm.call_llm import parse_llm_api_key

def get_embedding(embedding: str, embedding_key: str=None, env_file: str=None):
    if embedding == 'm3e':
        import os
        from langchain_huggingface import HuggingFaceEmbeddings
        # 优先使用本地路径，其次用镜像站下载
        local_model_path = r"E:\models\m3e-base"
        if embedding_key:
            model_path = embedding_key
        elif os.path.exists(local_model_path):
            model_path = local_model_path
        else:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            model_path = "moka-ai/m3e-base"
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    if embedding_key == None:
        embedding_key = parse_llm_api_key(embedding)
    if embedding == "openai":
        return OpenAIEmbeddings(openai_api_key=embedding_key)
    elif embedding == "zhipuai":
        return ZhipuAIEmbeddings(zhipuai_api_key=embedding_key)
    else:
        raise ValueError(f"embedding {embedding} not support ")
