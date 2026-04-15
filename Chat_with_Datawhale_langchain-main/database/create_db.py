#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
创建向量数据库
"""

import os
import sys
from typing import List, Dict, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_db_info(
    file_path: str,
    embedding_model: str = "m3e",
    persist_directory: Optional[str] = None,
    subject: str = "全部"
) -> str:
    """
    创建向量数据库

    Args:
        file_path: 文件或目录路径
        embedding_model: 嵌入模型名称 (m3e, openai, zhipuai)
        persist_directory: 向量数据库持久化路径
        subject: 学科名称

    Returns:
        创建结果信息
    """
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 设置默认持久化路径
    if persist_directory is None:
        persist_directory = os.path.join(base_dir, "exam_vector_db", "chroma")

    try:
        from langchain_community.document_loaders import (
            TextLoader,
            UnstructuredFileLoader,
            PyMuPDFLoader,
            DirectoryLoader
        )
        from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from embedding.call_embedding import get_embedding
        import shutil

        # 获取嵌入函数
        embeddings = get_embedding(embedding_model)

        # 如果存在旧的向量数据库，先删除
        if os.path.exists(persist_directory):
            print(f"删除旧的向量数据库: {persist_directory}")
            shutil.rmtree(persist_directory)

        # 确保目录存在
        os.makedirs(os.path.dirname(persist_directory), exist_ok=True)

        # 收集所有文档
        all_documents = []

        # 确定要处理的文件路径
        if os.path.isfile(file_path):
            # 单个文件
            files = [file_path]
        elif os.path.isdir(file_path):
            # 目录
            if subject == "全部":
                # 处理所有学科
                subject_dirs = ["政治", "英语", "数学", "控制工程"]
                files = []
                for subj in subject_dirs:
                    subj_dir = os.path.join(file_path, subj)
                    if os.path.exists(subj_dir):
                        # 获取目录下所有支持的文件
                        for root, _, filenames in os.walk(subj_dir):
                            for filename in filenames:
                                if filename.endswith(('.txt', '.md', '.pdf', '.docx')):
                                    files.append(os.path.join(root, filename))
            else:
                # 处理指定学科
                subject_dir = os.path.join(file_path, subject)
                if os.path.exists(subject_dir):
                    files = []
                    for root, _, filenames in os.walk(subject_dir):
                        for filename in filenames:
                            if filename.endswith(('.txt', '.md', '.pdf', '.docx')):
                                files.append(os.path.join(root, filename))
                else:
                    print(f"学科目录不存在: {subject_dir}")
                    return f"错误: 学科目录不存在 - {subject_dir}"
        else:
            return f"错误: 路径不存在 - {file_path}"

        print(f"找到 {len(files)} 个文件需要处理")

        # 加载文档
        for file_path in files:
            try:
                # 根据文件类型选择加载器
                if file_path.endswith('.pdf'):
                    loader = PyMuPDFLoader(file_path)
                elif file_path.endswith('.docx'):
                    loader = UnstructuredFileLoader(file_path, mode="elements")
                else:
                    loader = TextLoader(file_path, encoding='utf-8')

                documents = loader.load()

                # 添加元数据
                for doc in documents:
                    doc.metadata['source'] = os.path.basename(file_path)
                    doc.metadata['subject'] = subject

                all_documents.extend(documents)
                print(f"已加载: {os.path.basename(file_path)}")

            except Exception as e:
                print(f"加载文件失败 {file_path}: {str(e)}")

        if not all_documents:
            return "错误: 没有找到可处理的文档"

        print(f"总共加载了 {len(all_documents)} 个文档片段")

        # 文本分割
        # 使用 RecursiveCharacterTextSplitter 对中文更友好
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

        texts = text_splitter.split_documents(all_documents)
        print(f"分割成 {len(texts)} 个文本块")

        # 创建向量数据库
        print(f"正在创建向量数据库 (模型: {embedding_model})...")
        print("这可能需要几分钟时间，请耐心等待...")

        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        vectordb.persist()
        print(f"向量数据库已创建并保存到: {persist_directory}")

        return f"✓ 向量数据库创建成功!\n  - 文档数: {len(all_documents)}\n  - 文本块数: {len(texts)}\n  - 嵌入模型: {embedding_model}\n  - 学科: {subject}\n  - 保存路径: {persist_directory}"

    except Exception as e:
        error_msg = f"创建向量数据库失败: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg


def create_db(file_path: str, persist_path: str, embedding):
    """创建并持久化向量数据库"""
    from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    import glob

    docs = []
    if os.path.isfile(file_path):
        files = [file_path]
    else:
        files = glob.glob(os.path.join(file_path, "**", "*.txt"), recursive=True) + \
                glob.glob(os.path.join(file_path, "**", "*.pdf"), recursive=True)

    for f in files:
        try:
            loader = PyMuPDFLoader(f) if f.endswith(".pdf") else TextLoader(f, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"加载失败 {f}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)
    os.makedirs(os.path.dirname(persist_path) if os.path.dirname(persist_path) else persist_path, exist_ok=True)
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_path)
    vectordb.persist()
    return vectordb


def load_knowledge_db(persist_path: str, embedding):
    """加载已有向量数据库"""
    from langchain_community.vectorstores import Chroma
    return Chroma(persist_directory=persist_path, embedding_function=embedding)


def main():
    """测试函数"""
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exam_db_path = os.path.join(base_dir, "exam_knowledge_db")

    print("创建考研资料向量数据库...")
    print("=" * 50)

    if len(sys.argv) > 1:
        subject = sys.argv[1]
    else:
        subject = "全部"

    result = create_db_info(
        file_path=exam_db_path,
        embedding_model="m3e",
        subject=subject
    )

    print("\n" + result)


if __name__ == "__main__":
    main()