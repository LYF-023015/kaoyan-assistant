"""
PDF 解析模块
策略优先级：
  1. 云端 MinerU API（配置了 MINERU_API_URL 时优先，避免本地大模型下载）
  2. 本地 MinerU (magic_pdf)
  3. PyMuPDF 降级
"""
import os
import re
import time
import zipfile
import tempfile
import shutil
import requests
from pathlib import Path
from typing import List, Dict, Optional


class Document:
    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(content={self.content[:80]}..., metadata={self.metadata})"


class PDFParser:
    def __init__(self):
        self.cloud_mineru_url = os.getenv("MINERU_API_URL", "").strip()
        self.cloud_mineru_key = os.getenv("MINERU_API_KEY", "").strip()
        self.local_mineru_available = self._check_local_mineru()

    def _check_local_mineru(self) -> bool:
        try:
            import magic_pdf
            return True
        except ImportError:
            return False

    def parse(self, file_path: str, subject: str = "") -> List[Document]:
        """
        解析 PDF 文件，返回 Document 列表
        """
        # 优先使用云端 MinerU（无需本地大模型）
        if self.cloud_mineru_url and self.cloud_mineru_key:
            try:
                return self._parse_with_cloud_mineru(file_path, subject)
            except Exception as e:
                print(f"云端 MinerU 解析失败: {e}")
                # 云端失败时继续尝试本地

        # 本地 MinerU
        if self.local_mineru_available:
            try:
                return self._parse_with_local_mineru(file_path, subject)
            except Exception as e:
                print(f"本地 MinerU 解析失败，降级到 PyMuPDF: {e}")
                return self._parse_with_pymupdf(file_path, subject)
        else:
            return self._parse_with_pymupdf(file_path, subject)

    def _parse_with_cloud_mineru(self, file_path: str, subject: str) -> List[Document]:
        """
        调用 MinerU 云端 API 解析 PDF
        流程：申请上传URL -> PUT上传 -> 轮询结果 -> 下载ZIP -> 解压取MD
        """
        base_url = self.cloud_mineru_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {self.cloud_mineru_key}",
            "Content-Type": "application/json",
        }
        session = requests.Session()
        session.trust_env = False

        file_path_obj = Path(file_path)
        file_name = file_path_obj.name

        # 步骤1：申请上传 URL
        upload_payload = {"files": [{"name": file_name}], "model_version": "vlm"}
        resp = session.post(f"{base_url}/file-urls/batch", json=upload_payload, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if result.get("code") != 0:
            raise ValueError(f"获取上传链接失败: {result.get('msg')}")

        batch_id = result["data"]["batch_id"]
        upload_url = result["data"]["file_urls"][0]

        # 步骤2：PUT 上传文件
        with open(file_path, "rb") as f:
            put_resp = session.put(upload_url, data=f, timeout=60)
        put_resp.raise_for_status()

        # 步骤3：轮询解析结果
        result_url = f"{base_url}/extract-results/batch/{batch_id}"
        max_wait = 300  # 最多等待5分钟
        start_time = time.time()
        zip_url = None

        while time.time() - start_time < max_wait:
            resp = session.get(result_url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                raise ValueError(f"查询结果失败: {data.get('msg')}")

            extract_result = data["data"]["extract_result"][0]
            state = extract_result.get("state")

            if state == "done":
                zip_url = extract_result.get("full_zip_url")
                if zip_url:
                    break
                else:
                    raise ValueError("解析完成但未返回 ZIP 链接")
            elif state == "failed":
                err_msg = extract_result.get("err_msg", "未知错误")
                raise ValueError(f"MinerU 云端解析失败: {err_msg}")

            time.sleep(3)

        if not zip_url:
            raise TimeoutError("MinerU 云端解析超时")

        # 步骤4：下载 ZIP 并解压提取 Markdown
        zip_resp = session.get(zip_url, timeout=60)
        zip_resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(zip_resp.content)
            tmp_path = tmp.name

        extract_dir = Path(tempfile.mkdtemp())
        try:
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(extract_dir)

            # 查找 Markdown 文件
            md_files = list(extract_dir.rglob("*.md"))
            target_md = None
            # 优先找 full.md 或与原文件名同名的 md
            for md in md_files:
                if md.name.lower() == "full.md":
                    target_md = md
                    break
            if not target_md and md_files:
                target_md = md_files[0]

            if not target_md:
                raise ValueError("ZIP 中未找到 Markdown 文件")

            md_content = target_md.read_text(encoding="utf-8")
        finally:
            os.unlink(tmp_path)
            shutil.rmtree(extract_dir, ignore_errors=True)

        # 分块处理
        chunks = self._split_markdown(md_content, file_name)
        documents = []
        for idx, chunk in enumerate(chunks):
            documents.append(Document(
                content=chunk,
                metadata={
                    "source": file_name,
                    "subject": subject,
                    "file_type": "pdf",
                    "chunk_index": idx,
                    "parser": "cloud_mineru",
                }
            ))
        return documents

    def _parse_with_local_mineru(self, file_path: str, subject: str) -> List[Document]:
        """
        使用本地 MinerU 解析 PDF，提取结构化 Markdown
        """
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
        from magic_pdf.config.enums import SupportedPdfParseMethod

        name = Path(file_path).stem
        dir_path = Path(file_path).parent

        local_image_dir = dir_path / "images"
        local_image_dir.mkdir(exist_ok=True)
        image_dir = str(local_image_dir)
        image_writer = FileBasedDataWriter(image_dir)
        md_writer = FileBasedDataWriter(str(dir_path))
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(file_path)

        ds = PymuDocDataset(pdf_bytes)
        parse_method = ds.classify()

        if parse_method == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        md_content = pipe_result.get_markdown(str(local_image_dir))

        chunks = self._split_markdown(md_content, Path(file_path).name)
        documents = []
        for idx, chunk in enumerate(chunks):
            documents.append(Document(
                content=chunk,
                metadata={
                    "source": Path(file_path).name,
                    "subject": subject,
                    "file_type": "pdf",
                    "chunk_index": idx,
                    "parser": "local_mineru",
                }
            ))
        return documents

    def _parse_with_pymupdf(self, file_path: str, subject: str) -> List[Document]:
        """
        使用 PyMuPDF 解析 PDF，简单文本提取
        """
        import fitz

        doc = fitz.open(file_path)
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
        doc.close()

        full_text = "\n".join(text_parts)
        chunks = self._split_text(full_text, chunk_size=1200, overlap=200)
        documents = []
        for idx, chunk in enumerate(chunks):
            documents.append(Document(
                content=chunk,
                metadata={
                    "source": Path(file_path).name,
                    "subject": subject,
                    "file_type": "pdf",
                    "chunk_index": idx,
                    "parser": "pymupdf",
                }
            ))
        return documents

    def _split_markdown(self, md_content: str, source_name: str) -> List[str]:
        """
        按 Markdown 标题分块，保持公式和图片描述完整
        """
        sections = re.split(r'(?=\n#{2,3}\s)', md_content)
        chunks = []
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            if len(sec) > 2000:
                paragraphs = sec.split("\n\n")
                current = ""
                for para in paragraphs:
                    if len(current) + len(para) < 1500:
                        current += "\n\n" + para if current else para
                    else:
                        if current:
                            chunks.append(current)
                        current = para
                if current:
                    chunks.append(current)
            else:
                chunks.append(sec)
        return chunks or [md_content]

    def _split_text(self, text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
        """
        简单文本分块
        """
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            split_pos = text.rfind('。', end - 100, end)
            if split_pos == -1:
                split_pos = text.rfind('\n', end - 100, end)
            if split_pos == -1:
                split_pos = end
            chunks.append(text[start:split_pos + 1])
            start = split_pos + 1 - overlap
            if start < 0:
                start = 0
        return chunks
