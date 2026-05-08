"""
BM25 稀疏索引模块
使用 jieba 分词 + rank-bm25
"""
import os
import pickle
import jieba
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from backend.config import BM25_INDEX_PATH, SUBJECTS
from backend.core.document_processor import Document


class BM25Index:
    def __init__(self, index_path: str = None):
        self.index_path = str(index_path or BM25_INDEX_PATH)
        self.bm25 = None
        self.documents = []  # 存储 (text, metadata, doc_id)
        self.loaded = False

    def _tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut_for_search(text))

    def build(self, documents: List[Tuple[str, dict]]):
        """
        构建 BM25 索引
        documents: List[(text, metadata)]
        """
        self.documents = []
        tokenized_corpus = []
        for idx, (text, metadata) in enumerate(documents):
            if not text or not text.strip():
                continue
            self.documents.append({
                "id": idx,
                "text": text,
                "metadata": metadata,
            })
            tokens = self._tokenize(text)
            tokenized_corpus.append(tokens)

        if not tokenized_corpus:
            print("BM25: 无有效文档，跳过构建")
            return

        self.bm25 = BM25Okapi(tokenized_corpus)
        self.loaded = True
        self.save()
        print(f"BM25 索引构建完成，文档数: {len(tokenized_corpus)}")

    def search(self, query: str, k: int = 50, subject: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        BM25 检索，返回 (Document, bm25_score) 列表，按分数降序
        """
        if self.bm25 is None:
            if not self.load():
                return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 收集结果
        results = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            doc_info = self.documents[idx]
            # 学科过滤
            if subject and subject != "全部":
                doc_subject = doc_info["metadata"].get("subject", "")
                if doc_subject != subject:
                    continue
            doc = Document(
                content=doc_info["text"],
                metadata=doc_info["metadata"],
            )
            results.append((doc, float(score)))

        # 按分数降序，取 top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def save(self):
        """保存索引到磁盘"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "documents": self.documents,
            }, f)
        print(f"BM25 索引已保存到: {self.index_path}")

    def load(self) -> bool:
        """从磁盘加载索引"""
        if not os.path.exists(self.index_path):
            return False
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.documents = data["documents"]
            self.loaded = True
            print(f"BM25 索引已加载，文档数: {len(self.documents)}")
            return True
        except Exception as e:
            print(f"BM25 索引加载失败: {e}")
            return False


# 全局单例
_bm25_index = None


def get_bm25_index() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index
