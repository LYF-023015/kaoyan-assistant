"""
FastAPI 路由定义
"""
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.core.react_agent import get_agent
from backend.core.document_processor import DocumentProcessor
from backend.core.vector_store import get_vector_store
from backend.core.bm25_index import get_bm25_index
from backend.llm import get_llm_client
from backend.config import KNOWLEDGE_BASE_DIR, SUBJECTS, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_DIR

router = APIRouter()

# 增量索引状态文件
INDEX_STATE_PATH = VECTOR_DB_DIR / "index_state.json"


# ========== 数据模型 ==========
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    subject: Optional[str] = "全部"
    use_web_search: Optional[bool] = True
    model: Optional[str] = None
    temperature: Optional[float] = None


class IndexRequest(BaseModel):
    subject: Optional[str] = "全部"
    force: Optional[bool] = False


# ========== 索引状态管理 ==========
def _load_index_state() -> dict:
    if INDEX_STATE_PATH.exists():
        try:
            with open(INDEX_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_index_state(state: dict):
    try:
        with open(INDEX_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存索引状态失败: {e}")


def _rebuild_state_from_chroma(vector_store, subject: str) -> dict:
    """
    当 index_state.json 丢失时，从 ChromaDB 中重建该学科的 state。
    按 source 文件分组，统计每个文件的 chunks 数和 doc_ids。
    """
    from backend.config import KNOWLEDGE_BASE_DIR

    collection = vector_store._get_collection(subject)
    count = collection.count()
    if count == 0:
        return {}

    print(f"[{subject}] 状态文件丢失，从 ChromaDB 重建 state ({count} 个文档)...")
    state = {}

    # 分批获取所有文档的 id 和 metadata
    batch_size = 500
    for offset in range(0, count, batch_size):
        results = collection.get(
            limit=min(batch_size, count - offset),
            offset=offset,
            include=["metadatas"],
        )
        ids = results.get("ids", []) or []
        metadatas = results.get("metadatas", []) or []

        for doc_id, meta in zip(ids, metadatas):
            source = meta.get("source", "")
            if not source:
                continue
            if source not in state:
                state[source] = {"chunks": 0, "doc_ids": []}
            state[source]["chunks"] += 1
            state[source]["doc_ids"].append(doc_id)

    # 尝试从文件系统补全 mtime/size
    subj_dir = KNOWLEDGE_BASE_DIR / subject
    for source_name, info in state.items():
        file_path = subj_dir / source_name
        if file_path.exists():
            stat = file_path.stat()
            info["mtime"] = stat.st_mtime
            info["size"] = stat.st_size

    return state


# ========== 聊天接口 ==========
@router.post("/chat")
async def chat(req: ChatRequest):
    loop = asyncio.get_event_loop()
    agent = get_agent()

    def _run():
        return agent.run(
            query=req.message,
            subject=req.subject,
            use_web_search=req.use_web_search,
            history=req.history,
        )

    answer, sources = await loop.run_in_executor(None, _run)
    return {
        "answer": answer,
        "sources": sources,
    }


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def event_generator():
        loop = asyncio.get_event_loop()

        def _retrieve():
            from backend.core.retriever import get_retriever
            retriever = get_retriever()
            return retriever.retrieve(req.message, subject=req.subject, use_reranker=True)

        retrieved = await loop.run_in_executor(None, _retrieve)

        context_parts = []
        sources = []
        for doc, score in retrieved:
            meta = doc.metadata
            context_parts.append(
                f"[来源: {meta.get('source', '未知')} 分数: {score:.4f}]\n{doc.content}"
            )
            sources.append({
                "content": doc.content[:300],
                "metadata": meta,
                "score": score,
            })
        context = "\n\n---\n\n".join(context_parts)

        web_context = ""
        if req.use_web_search and len(retrieved) < 3:
            def _search():
                from backend.core.web_search import web_search
                return web_search(req.message, max_results=3)
            web_context = await loop.run_in_executor(None, _search)

        system_prompt = f"""你是一个专业的考研辅导助手。请根据以下资料回答问题。

【本地知识库资料】
{context}

【网络搜索资料】
{web_context}

要求：
- 优先引用本地知识库的内容，准确作答
- 如果涉及数学公式，使用 LaTeX 格式
- 分点阐述，保持简明扼要
- 在回答末尾标注引用来源
- 如果信息不足，明确告知用户
"""

        messages = [{"role": "system", "content": system_prompt}]
        if req.history:
            for h in req.history[-6:]:
                messages.append(h)
        messages.append({"role": "user", "content": req.message})

        llm = get_llm_client()
        full_answer = ""

        def generate():
            for token in llm.chat_stream(messages):
                yield token

        def _next_token(gen):
            try:
                return next(gen)
            except StopIteration:
                return None

        gen = generate()
        while True:
            try:
                token = await loop.run_in_executor(None, _next_token, gen)
                if token is None:
                    break
                full_answer += token
                data = token.replace("\n", "\\n").replace('"', '\\"')
                yield f'data: {{"token": "{data}"}}\n\n'
            except Exception as e:
                yield f'data: {{"error": "{str(e)}"}}\n\n'
                break

        yield f'data: {{"done": true, "answer": {json.dumps(full_answer, ensure_ascii=False)}, "sources": {json.dumps(sources, ensure_ascii=False)}}}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ========== 文件上传 ==========
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    subject: str = Form("全部"),
):
    if subject not in SUBJECTS and subject != "全部":
        return {"success": False, "message": f"无效的学科: {subject}"}

    save_dir = KNOWLEDGE_BASE_DIR / subject if subject != "全部" else KNOWLEDGE_BASE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / file.filename
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {
            "success": True,
            "message": f"文件上传成功: {file.filename}",
            "path": str(file_path),
            "filename": file.filename,
            "subject": subject,
        }
    except Exception as e:
        return {"success": False, "message": f"上传失败: {str(e)}"}


# ========== 知识库文件列表 ==========
@router.get("/files")
async def get_files(subject: Optional[str] = None):
    files = []
    subjects_to_scan = [subject] if subject and subject != "全部" else SUBJECTS

    for subj in subjects_to_scan:
        subj_dir = KNOWLEDGE_BASE_DIR / subj
        if not subj_dir.exists():
            continue
        for file_path in subj_dir.rglob("*"):
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "subject": subj,
                    "path": str(file_path.relative_to(KNOWLEDGE_BASE_DIR)),
                    "size": file_path.stat().st_size,
                })

    return {
        "files": files,
        "total": len(files),
    }


# ========== 建库接口（纯增量，不清空已有数据） ==========
@router.post("/index")
async def build_index(req: IndexRequest):
    async def event_generator():
        loop = asyncio.get_event_loop()

        def _do_build():
            processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            vector_store = get_vector_store()
            state = _load_index_state()

            subjects_to_index = [req.subject] if req.subject != "全部" else SUBJECTS

            # 强制重建：清空所有数据和状态
            if req.force:
                for subj in subjects_to_index:
                    vector_store.clear_subject(subj)
                    if subj in state:
                        del state[subj]
                yield {
                    "type": "progress",
                    "status": "force",
                    "message": "强制重建模式：清空所有旧索引并重新处理",
                }

            total_new_chunks = 0
            processed_details = []
            needs_bm25_rebuild = False

            for subj in subjects_to_index:
                subj_dir = KNOWLEDGE_BASE_DIR / subj

                # 如果该学科不在 state 中，但 ChromaDB 已有数据，尝试重建 state
                if subj not in state:
                    chroma_state = _rebuild_state_from_chroma(vector_store, subj)
                    if chroma_state:
                        state[subj] = chroma_state

                if not subj_dir.exists():
                    # 目录不存在但 state 中有数据，清理残留索引
                    if subj in state and not req.force:
                        old_items = list(state[subj].items())
                        for old_name, old_info in old_items:
                            doc_ids = old_info.get("doc_ids", [])
                            if doc_ids:
                                vector_store.delete_by_ids(doc_ids, subject=subj)
                        del state[subj]
                        needs_bm25_rebuild = True
                        yield {
                            "type": "progress",
                            "subject": subj,
                            "status": "cleaned",
                            "message": f"[{subj}] 目录已删除，清理了 {len(old_items)} 个文件的索引",
                        }
                    continue

                # 收集当前目录中的文件
                files = []
                for f in subj_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
                        files.append(f)

                current_filenames = {f.name for f in files}
                old_state = state.get(subj, {})

                # 分类文件
                files_to_process = []
                files_unchanged = []

                if req.force:
                    files_to_process = files
                else:
                    for f in files:
                        stat = f.stat()
                        file_key = f.name
                        if file_key in old_state:
                            old_info = old_state[file_key]
                            if old_info.get("mtime") == stat.st_mtime and old_info.get("size") == stat.st_size:
                                files_unchanged.append(f)
                                continue
                        files_to_process.append(f)

                # 清理已删除的文件索引（只在非 force 模式下）
                deleted_count = 0
                if not req.force:
                    for old_name, old_info in list(old_state.items()):
                        if old_name not in current_filenames:
                            doc_ids = old_info.get("doc_ids", [])
                            if doc_ids:
                                vector_store.delete_by_ids(doc_ids, subject=subj)
                            del state[subj][old_name]
                            deleted_count += 1

                if deleted_count > 0:
                    needs_bm25_rebuild = True
                    yield {
                        "type": "progress",
                        "subject": subj,
                        "status": "cleaned",
                        "message": f"[{subj}] 清理了 {deleted_count} 个已删除文件的索引",
                    }

                # 如果没有文件且 state 已空
                if not files and not old_state:
                    continue

                # 如果没有需要处理的文件
                if not files_to_process:
                    subj_total = sum(info.get("chunks", 0) for info in state.get(subj, {}).values())
                    yield {
                        "type": "progress",
                        "subject": subj,
                        "status": "skipped",
                        "message": f"[{subj}] {len(files_unchanged)} 个文件已索引，无需更新",
                    }
                    processed_details.append({
                        "subject": subj,
                        "files": list(state.get(subj, {}).keys()),
                        "new_chunks": 0,
                        "total_chunks": subj_total,
                    })
                    continue

                yield {
                    "type": "progress",
                    "subject": subj,
                    "status": "started",
                    "files": [f.name for f in files_to_process],
                    "message": f"[{subj}] 发现 {len(files_to_process)} 个新/修改的文件" + (
                        f"，跳过 {len(files_unchanged)} 个未变更文件" if files_unchanged else ""
                    ),
                }

                subject_new_chunks = 0

                for idx, file_path in enumerate(files_to_process):
                    yield {
                        "type": "progress",
                        "subject": subj,
                        "status": "file",
                        "current": idx + 1,
                        "total": len(files_to_process),
                        "current_file": file_path.name,
                        "message": f"[{subj}] ({idx + 1}/{len(files_to_process)}) 正在解析: {file_path.name}",
                    }

                    docs = processor.process_file(str(file_path), subject=subj)
                    chunk_count = len(docs)
                    if docs:
                        doc_ids = vector_store.add_documents(docs, subject=subj)
                        stat = file_path.stat()
                        if subj not in state:
                            state[subj] = {}
                        state[subj][file_path.name] = {
                            "mtime": stat.st_mtime,
                            "size": stat.st_size,
                            "doc_ids": doc_ids,
                            "chunks": chunk_count,
                        }
                        total_new_chunks += chunk_count
                        subject_new_chunks += chunk_count

                    yield {
                        "type": "progress",
                        "subject": subj,
                        "status": "file_done",
                        "current": idx + 1,
                        "total": len(files_to_process),
                        "current_file": file_path.name,
                        "chunks": chunk_count,
                        "message": f"[{subj}] ({idx + 1}/{len(files_to_process)}) 完成: {file_path.name} → {chunk_count} 个文本块",
                    }

                if subject_new_chunks > 0:
                    needs_bm25_rebuild = True

                subj_total_chunks = sum(info.get("chunks", 0) for info in state.get(subj, {}).values())
                processed_details.append({
                    "subject": subj,
                    "files": list(state.get(subj, {}).keys()),
                    "new_chunks": subject_new_chunks,
                    "total_chunks": subj_total_chunks,
                })

                yield {
                    "type": "progress",
                    "subject": subj,
                    "status": "done",
                    "new_chunks": subject_new_chunks,
                    "total_chunks": subj_total_chunks,
                    "files": list(state.get(subj, {}).keys()),
                    "message": f"[{subj}] 完成，新增 {subject_new_chunks} 个文本块，总计 {subj_total_chunks} 个",
                }

            # 重建 BM25（只在有变更时）
            if needs_bm25_rebuild or req.force:
                yield {"type": "progress", "status": "bm25", "message": "正在更新 BM25 索引..."}
                bm25 = get_bm25_index()
                bm25_docs = []
                for subj in subjects_to_index:
                    docs = vector_store.get_all_documents_texts(subject=subj)
                    bm25_docs.extend(docs)
                bm25.build(bm25_docs)

            _save_index_state(state)

            if total_new_chunks == 0 and not needs_bm25_rebuild and not req.force:
                yield {
                    "type": "done",
                    "success": True,
                    "message": "所有文件已是最新索引状态，无需更新",
                    "subjects": subjects_to_index,
                    "details": processed_details,
                    "total_chunks": 0,
                }
            else:
                yield {
                    "type": "done",
                    "success": True,
                    "message": f"索引更新完成！新增 {total_new_chunks} 个文本块",
                    "subjects": subjects_to_index,
                    "details": processed_details,
                    "total_chunks": total_new_chunks,
                }

        gen = _do_build()
        while True:
            try:
                data = await loop.run_in_executor(None, next, gen)
                json_str = json.dumps(data, ensure_ascii=False).replace("\n", "\\n")
                yield f'data: {json_str}\n\n'
                if data.get("type") == "done":
                    break
            except StopIteration:
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                err = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
                yield f'data: {err}\n\n'
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


# ========== 学科列表 ==========
@router.get("/subjects")
async def get_subjects():
    return {"subjects": SUBJECTS}
