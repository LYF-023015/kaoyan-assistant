"""
Embedding 封装：M3E 稠密向量
"""
import os
import torch
from typing import List
from sentence_transformers import SentenceTransformer


class M3EEmbedding:
    def __init__(self, model_path: str = None, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 模型路径优先级：传入 > 本地路径 > HuggingFace 镜像
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        else:
            local_path = r"E:\models\m3e-base"
            if os.path.exists(local_path):
                self.model_path = local_path
            else:
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                self.model_path = "moka-ai/m3e-base"

        print(f"Loading M3E model from {self.model_path} on {self.device} ...")
        self.model = SentenceTransformer(self.model_path, device=self.device)
        self.model.eval()
        self.dimension = self.model.get_embedding_dimension()
        print(f"M3E loaded. Dimension: {self.dimension}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量文档嵌入，返回归一化向量
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        查询嵌入
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()


# 全局单例
_embedding_model = None


def get_embedding_model() -> M3EEmbedding:
    global _embedding_model
    if _embedding_model is None:
        from backend.config import M3E_MODEL_PATH, DEVICE
        _embedding_model = M3EEmbedding(model_path=M3E_MODEL_PATH, device=DEVICE)
    return _embedding_model
