"""
融合检索模块
M3E 稠密召回 + BM25 稀疏召回 -> RRF 融合 -> BGE-Reranker -> TOP10
"""
import torch
from typing import List, Tuple, Optional
from backend.core.vector_store import get_vector_store
from backend.core.bm25_index import get_bm25_index
from backend.core.document_processor import Document
from backend.config import DENSE_TOP_K, SPARSE_TOP_K, RERANK_TOP_K, FINAL_TOP_K, RRF_K, BGE_RERANKER_PATH, DEVICE


class BGEReranker:
    def __init__(self, model_path: str = None, device: str = "auto"):
        self.model_path = model_path or BGE_RERANKER_PATH
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = None
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            self._load_model()
            self._loaded = True

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import os
            # 设置镜像站，加速下载
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            print(f"Loading BGE-Reranker from {self.model_path} on {self.device} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, local_files_only=False)
            self.model.to(self.device)
            self.model.eval()
            print("BGE-Reranker loaded.")
        except Exception as e:
            print(f"BGE-Reranker 加载失败（将不使用重排序）: {e}")
            self.tokenizer = None
            self.model = None

    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        对文档列表进行重排序
        """
        self._ensure_loaded()
        if self.model is None or not documents:
            return [(doc, 0.0) for doc in documents[:top_k]]

        pairs = [[query, doc.content] for doc in documents]
        scores = []

        batch_size = 8
        from torch.nn.functional import softmax

        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # 取 logits 的 softmax 正类概率作为相关性分数
                batch_scores = softmax(outputs.logits, dim=-1)[:, 1].cpu().tolist()
                scores.extend(batch_scores)

        # 按分数排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]


class FusionRetriever:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.bm25_index = get_bm25_index()
        self.reranker = BGEReranker()

    def retrieve(
        self,
        query: str,
        subject: Optional[str] = None,
        use_reranker: bool = True,
    ) -> List[Tuple[Document, float]]:
        """
        完整检索流程：
        1. 稠密召回 (M3E + Chroma)
        2. 稀疏召回 (BM25)
        3. RRF 融合
        4. BGE-Reranker 重排序（可选）
        返回 TOP10 结果
        """
        # 1. 稠密召回
        dense_results = self.vector_store.similarity_search(query, k=DENSE_TOP_K, subject=subject)

        # 2. 稀疏召回
        sparse_results = self.bm25_index.search(query, k=SPARSE_TOP_K, subject=subject)

        # 3. RRF 融合
        fused_results = self._rrf_fusion(dense_results, sparse_results)

        if not fused_results:
            return []

        # 4. Reranker
        if use_reranker and self.reranker.model is not None:
            # 取 RRF 前 RERANK_TOP_K 名送入 Reranker
            docs_to_rerank = [doc for doc, _ in fused_results[:RERANK_TOP_K]]
            reranked = self.reranker.rerank(query, docs_to_rerank, top_k=FINAL_TOP_K)
            return reranked
        else:
            return fused_results[:FINAL_TOP_K]

    def _rrf_fusion(
        self,
        dense_results: List[Tuple[Document, float]],
        sparse_results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Reciprocal Rank Fusion
        score = sum(1 / (k + rank)) for each list
        """
        k = RRF_K
        doc_scores = {}
        doc_map = {}

        # 处理稠密召回结果
        for rank, (doc, score) in enumerate(dense_results, start=1):
            key = self._doc_key(doc)
            doc_map[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank)

        # 处理稀疏召回结果
        for rank, (doc, score) in enumerate(sparse_results, start=1):
            key = self._doc_key(doc)
            doc_map[key] = doc
            doc_scores[key] = doc_scores.get(key, 0.0) + 1.0 / (k + rank)

        # 排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[key], rrf_score) for key, rrf_score in sorted_docs]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """生成文档唯一键"""
        meta = doc.metadata
        return f"{meta.get('source', '')}_{meta.get('chunk_index', 0)}_{hash(doc.content) & 0xFFFFFF}"


# 全局单例
_retriever = None


def get_retriever() -> FusionRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FusionRetriever()
    return _retriever
