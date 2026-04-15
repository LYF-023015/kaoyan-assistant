#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
文档处理器
处理各种格式的文档并生成向量化知识库
"""

import os
import re
import hashlib
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """文档处理器类"""

    def __init__(self, embedding_model="m3e", chunk_size=1000, chunk_overlap=200):
        """
        初始化文档处理器

        Args:
            embedding_model: 使用的嵌入模型
            chunk_size: 文档分块大小
            chunk_overlap: 文档分块重叠大小
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 支持的文件格式
        self.supported_formats = {
            '.txt': self._process_text_file,
            '.md': self._process_markdown_file,
            '.pdf': self._process_pdf_file,
            '.docx': self._process_docx_file
        }

        # 导入必要的模块
        self._import_required_modules()

        # 导入嵌入模块
        try:
            from embedding.call_embedding import get_embedding
            self.embedding_func = get_embedding(embedding_model)
            logger.info(f"使用嵌入模型: {embedding_model}")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            self.embedding_func = None

    def _import_required_modules(self):
        """导入必要的模块"""
        try:
            import fitz  # 用于PDF处理
            self.fitz = fitz
        except ImportError:
            logger.warning("PyMuPDF未安装，将无法处理PDF文件")
            self.fitz = None

        try:
            from docx import Document  # 用于Word文档处理
            self.docx = Document
        except ImportError:
            logger.warning("python-docx未安装，将无法处理Word文档")
            self.docx = None

    def _process_text_file(self, file_path: str) -> List[Dict]:
        """处理文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self._split_text(content, os.path.basename(file_path))

    def _process_markdown_file(self, file_path: str) -> List[Dict]:
        """处理Markdown文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 简单的Markdown处理：去除Markdown语法，保留文本内容
        content = re.sub(r'#{1,6}\s*', '', content)  # 去除标题
        content = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', content)  # 去除强调
        content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)  # 去除链接

        return self._split_text(content, os.path.basename(file_path))

    def _process_pdf_file(self, file_path: str) -> List[Dict]:
        """处理PDF文件"""
        if not self.fitz:
            logger.error("未安装PyMuPDF，无法处理PDF文件")
            return []

        doc = self.fitz.open(file_path)
        text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(text)

        full_text = '\n'.join(text_parts)
        return self._split_text(full_text, os.path.basename(file_path))

    def _process_docx_file(self, file_path: str) -> List[Dict]:
        """处理Word文档"""
        if not self.docx:
            logger.error("未安装python-docx，无法处理Word文档")
            return []

        doc = self.docx(file_path)
        paragraphs = []

        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)

        full_text = '\n'.join(paragraphs)
        return self._split_text(full_text, os.path.basename(file_path))

    def _split_text(self, text: str, filename: str) -> List[Dict]:
        """分割文本为多个块"""
        # 清理文本
        text = re.sub(r'\s+', ' ', text).strip()

        # 如果文本较短，直接返回
        if len(text) <= self.chunk_size:
            return [{
                'content': text,
                'metadata': {
                    'source': filename,
                    'chunk_id': 0,
                    'total_chunks': 1,
                    'file_size': len(text)
                }
            }]

        # 分割文本
        chunks = []
        current_chunk = ""
        current_size = 0

        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            if current_size + len(sentence) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " "
                current_chunk += sentence
                current_size = len(current_chunk)
            else:
                # 保存当前块
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': {
                            'source': filename,
                            'chunk_id': len(chunks),
                            'total_chunks': (len(text) + self.chunk_size - self.chunk_overlap) // (self.chunk_size - self.chunk_overlap),
                            'file_size': len(current_chunk)
                        }
                    })

                    # 开始新块，允许重叠
                    overlap_text = current_chunk[-self.chunk_overlap:] if current_chunk else ""
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                    current_size = len(current_chunk)

        # 添加最后一个块
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': {
                    'source': filename,
                    'chunk_id': len(chunks),
                    'total_chunks': len(chunks),
                    'file_size': len(current_chunk)
                }
            })

        return chunks

    def process_directory(self, directory_path: str, subject: str = "全部") -> List[Dict]:
        """
        处理目录中的所有文档

        Args:
            directory_path: 目录路径
            subject: 学科名称

        Returns:
            处理后的文档块列表
        """
        if not os.path.exists(directory_path):
            logger.error(f"目录不存在: {directory_path}")
            return []

        all_chunks = []
        file_count = 0

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                if file_ext in self.supported_formats:
                    logger.info(f"正在处理文件: {file_path}")

                    try:
                        processor = self.supported_formats[file_ext]
                        chunks = processor(file_path)

                        # 添加学科信息到元数据
                        for chunk in chunks:
                            chunk['metadata']['subject'] = subject
                            chunk['metadata']['processed_time'] = datetime.now().isoformat()

                        all_chunks.extend(chunks)
                        file_count += 1
                        logger.info(f"文件 {file} 处理完成，生成了 {len(chunks)} 个块")

                    except Exception as e:
                        logger.error(f"处理文件 {file} 时出错: {e}")

        logger.info(f"共处理 {file_count} 个文件，生成 {len(all_chunks)} 个文档块")
        return all_chunks

    def create_embeddings(self, chunks: List[Dict]) -> Tuple[List[Dict], List[List[float]]]:
        """
        为文档块创建嵌入向量

        Args:
            chunks: 文档块列表

        Returns:
            (chunks_with_embeddings, embeddings)
        """
        if not self.embedding_func:
            logger.error("嵌入函数未初始化")
            return [], []

        logger.info("开始创建嵌入向量...")
        embeddings = []
        chunks_with_embeddings = []

        for i, chunk in enumerate(chunks):
            try:
                # 创建嵌入
                embedding = self.embedding_func.embed_documents([chunk['content']])

                if embedding and len(embedding) > 0:
                    chunk['embedding'] = embedding[0]
                    chunks_with_embeddings.append(chunk)
                    embeddings.append(embedding[0])

                    logger.info(f"已处理 {i+1}/{len(chunks)} 个块")
                else:
                    logger.warning(f"第 {i+1} 个块的嵌入创建失败")

            except Exception as e:
                logger.error(f"为第 {i+1} 个块创建嵌入时出错: {e}")

        logger.info(f"成功创建 {len(embeddings)} 个嵌入向量")
        return chunks_with_embeddings, embeddings

    def save_processed_data(self, chunks: List[Dict], output_dir: str) -> str:
        """
        保存处理后的数据

        Args:
            chunks: 带有嵌入的文档块
            output_dir: 输出目录

        Returns:
            保存的JSON文件路径
        """
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"processed_data_{timestamp}.json")

        # 保存数据
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"数据已保存到: {output_file}")
        return output_file

    def get_file_hash(self, file_path: str) -> str:
        """获取文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_statistics(self, chunks: List[Dict]) -> Dict:
        """获取处理统计信息"""
        stats = {
            'total_chunks': len(chunks),
            'subjects': {},
            'sources': {},
            'avg_chunk_size': 0,
            'total_content_length': 0
        }

        if chunks:
            # 统计学科分布
            for chunk in chunks:
                subject = chunk['metadata'].get('subject', '未知')
                stats['subjects'][subject] = stats['subjects'].get(subject, 0) + 1

                # 统计来源分布
                source = chunk['metadata'].get('source', '未知')
                stats['sources'][source] = stats['sources'].get(source, 0) + 1

                # 计算总长度和平均长度
                stats['total_content_length'] += len(chunk['content'])

            stats['avg_chunk_size'] = stats['total_content_length'] / stats['total_chunks']

        return stats


def main():
    """主函数示例"""
    # 创建文档处理器
    processor = DocumentProcessor()

    # 处理考试资料目录
    exam_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exam_knowledge_db")

    # 处理所有学科
    all_chunks = []
    for subject in ["政治", "英语", "数学", "控制工程"]:
        subject_path = os.path.join(exam_db_path, subject)
        if os.path.exists(subject_path):
            chunks = processor.process_directory(subject_path, subject)
            all_chunks.extend(chunks)

    # 创建嵌入
    chunks_with_embeddings, embeddings = processor.create_embeddings(all_chunks)

    # 保存数据
    if chunks_with_embeddings:
        output_dir = os.path.join(exam_db_path, "processed")
        saved_file = processor.save_processed_data(chunks_with_embeddings, output_dir)

        # 显示统计信息
        stats = processor.get_statistics(chunks_with_embeddings)
        print("\n处理统计信息:")
        print(f"总文档块数: {stats['total_chunks']}")
        print(f"平均块大小: {stats['avg_chunk_size']:.2f} 字符")
        print("\n学科分布:")
        for subject, count in stats['subjects'].items():
            print(f"  {subject}: {count} 个块")

        print(f"\n数据已保存到: {saved_file}")


if __name__ == "__main__":
    main()