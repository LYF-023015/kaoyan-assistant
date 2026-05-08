"""
ChromaDB 向量存储封装
按学科隔离 collection，使用 cosine 距离
"""
import os
import uuid
from typing import List, Tuple, Optional
import chromadb
from chromadb.config import Settings
from backend.core.embeddings import get_embedding_model
from backend.core.document_processor import Document
from backend.config import CHROMA_PERSIST_DIR, SUBJECTS, SUBJECT_NAME_MAP


class VectorStore:
    def __init__(self, persist_dir: str = None):
        self.persist_dir = str(persist_dir or CHROMA_PERSIST_DIR)
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedding_model = get_embedding_model()
        self._collections = {}

    def _get_collection(self, subject: str):
        """获取或创建学科对应的 collection"""
        if subject in self._collections:
            return self._collections[subject]

        collection_name = f"exam_{SUBJECT_NAME_MAP.get(subject, subject)}" if subject else "exam_default"
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collections[subject] = collection
        return collection

    def add_documents(self, documents: List[Document], subject: str = "") -> List[str]:
        """
        批量添加文档到向量库，返回添加的 doc_ids
        """
        if not documents:
            return []

        collection = self._get_collection(subject)
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        ids = [str(uuid.uuid4()) for _ in documents]
        metadatas = [doc.metadata for doc in documents]

        # 分批添加，避免单次过大
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end = i + batch_size
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 50,
        subject: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """
        稠密向量相似度检索，返回 (Document, score) 列表
        score 为 cosine similarity（越大越相似）
        """
        query_embedding = self.embedding_model.embed_query(query)

        if subject and subject != "全部":
            collection = self._get_collection(subject)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
            return self._format_results(results)
        else:
            # 跨学科搜索，合并所有 collection 的结果
            all_results = []
            for subj in [SUBJECT_NAME_MAP.get(s, s) for s in SUBJECTS]:
                try:
                    collection = self._get_collection(subj)
                    count = collection.count()
                    if count == 0:
                        continue
                    n = min(k, count)
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n,
                        include=["documents", "metadatas", "distances"],
                    )
                    all_results.extend(self._format_results(results))
                except Exception as e:
                    print(f"检索学科 {subj} 失败: {e}")
            # 按 score 排序，取 top k
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results[:k]

    def _format_results(self, results) -> List[Tuple[Document, float]]:
        """
        将 chromadb 结果格式化为 (Document, cosine_similarity)
        chromadb cosine distance = 1 - cosine_similarity
        """
        formatted = []
        if not results or not results["ids"]:
            return formatted

        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        for i in range(len(ids)):
            doc = Document(
                content=documents[i] if i < len(documents) else "",
                metadata=metadatas[i] if i < len(metadatas) else {},
            )
            # cosine distance -> cosine similarity
            distance = distances[i] if i < len(distances) else 1.0
            similarity = 1.0 - distance
            formatted.append((doc, similarity))
        return formatted

    def clear_subject(self, subject: str):
        """清空某个学科的 collection"""
        collection_name = f"exam_{SUBJECT_NAME_MAP.get(subject, subject)}" if subject else "exam_default"
        try:
            self.client.delete_collection(name=collection_name)
            if subject in self._collections:
                del self._collections[subject]
        except Exception as e:
            print(f"清空 collection 失败: {e}")

    def delete_by_ids(self, ids: List[str], subject: str = ""):
        """按 ID 删除文档"""
        if not ids:
            return
        collection = self._get_collection(subject)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            try:
                collection.delete(ids=ids[i:i + batch_size])
            except Exception as e:
                print(f"删除文档失败: {e}")

    def get_all_documents_texts(self, subject: Optional[str] = None) -> List[Tuple[str, dict]]:
        """
        获取所有文档的文本和元数据，用于构建 BM25 索引
        返回 [(text, metadata), ...]
        """
        all_docs = []
        subjects_to_query = [subject] if subject and subject != "全部" else SUBJECTS
        subjects_to_query = [SUBJECT_NAME_MAP.get(s, s) for s in subjects_to_query]
        for subj in subjects_to_query:
            try:
                collection = self._get_collection(subj)
                count = collection.count()
                if count == 0:
                    continue
                # 分批获取
                batch_size = 500
                for offset in range(0, count, batch_size):
                    results = collection.get(
                        limit=min(batch_size, count - offset),
                        offset=offset,
                        include=["documents", "metadatas"],
                    )
                    docs = results.get("documents", []) or []
                    metas = results.get("metadatas", []) or []
                    for d, m in zip(docs, metas):
                        if d:
                            all_docs.append((d, m))
            except Exception as e:
                print(f"获取学科 {subj} 文档失败: {e}")
        return all_docs


# 全局单例
_vector_store = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
