"""
从向量库中采样文档 chunks，用 LLM 自动生成测试 QA pairs。
每个 QA pair 包含：
  - query: 问题
  - answer: 标准答案
  - ground_truth_chunk_key: 源 chunk 的唯一标识（用于检索指标计算）
  - subject: 所属学科
  - source: 源文件名
"""
import json
import random
import hashlib
import os
from pathlib import Path
from typing import List, Dict

from backend.core.vector_store import get_vector_store
from backend.core.retriever import FusionRetriever
from backend.llm import get_llm_client
from backend.config import SUBJECTS, VECTOR_DB_DIR


TESTSET_DIR = Path(__file__).parent / "testset"
TESTSET_DIR.mkdir(parents=True, exist_ok=True)


def _doc_key(content: str, metadata: dict) -> str:
    """与 FusionRetriever._doc_key 保持一致的唯一键"""
    return f"{metadata.get('source', '')}_{metadata.get('chunk_index', 0)}_{hash(content) & 0xFFFFFF}"


def generate_qa_from_chunk(chunk_text: str, metadata: dict, llm, num_questions: int = 1) -> List[Dict]:
    """
    用一个文档 chunk 生成 QA pairs。
    返回 [{query, answer}, ...]
    """
    source = metadata.get("source", "未知文件")
    subject = metadata.get("subject", "")

    prompt = f"""你是一个考研命题专家。请根据以下教材片段，生成 {num_questions} 个高质量的考研复习问题及其标准答案。

【教材片段来源】{source}
【学科】{subject}

【教材内容】
{chunk_text[:1500]}

要求：
- 问题必须是该片段能够直接回答的，不要超出片段范围
- 答案必须忠实于教材内容
- 问题类型可以是概念解释、公式推导、方法总结、性质判断等
- 输出严格的 JSON 数组格式，不要有任何其他内容：

[
  {{"query": "问题1", "answer": "答案1"}},
  {{"query": "问题2", "answer": "答案2"}}
]
"""

    try:
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        # 尝试解析 JSON
        text = response.strip()
        # 有时候 LLM 会包裹在 markdown 代码块中
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        qa_list = json.loads(text)
        if isinstance(qa_list, dict):
            qa_list = [qa_list]
        return qa_list
    except Exception as e:
        print(f"  生成 QA 失败: {e}")
        return []


def build_testset(
    samples_per_subject: int = 15,
    questions_per_chunk: int = 1,
    seed: int = 42,
    output_name: str = "testset.json",
) -> Path:
    """
    从向量库采样 chunks，生成测试集。
    返回保存的文件路径。
    """
    random.seed(seed)
    vector_store = get_vector_store()
    llm = get_llm_client()

    test_cases = []
    total_generated = 0

    for subject in SUBJECTS:
        print(f"\n[{subject}] 采样文档 chunks...")
        docs = vector_store.get_all_documents_texts(subject=subject)
        if not docs:
            print(f"  [{subject}] 向量库为空，跳过")
            continue

        # 过滤掉过短的 chunk（至少 200 字符才有信息量）
        valid_docs = [(text, meta) for text, meta in docs if len(text.strip()) >= 200]
        if not valid_docs:
            print(f"  [{subject}] 没有有效文档，跳过")
            continue

        # 随机采样
        n_samples = min(samples_per_subject, len(valid_docs))
        sampled = random.sample(valid_docs, n_samples)
        print(f"  [{subject}] 从 {len(valid_docs)} 个 chunks 中采样 {n_samples} 个")

        for text, meta in sampled:
            qa_pairs = generate_qa_from_chunk(text, meta, llm, num_questions=questions_per_chunk)
            for qa in qa_pairs:
                test_cases.append({
                    "query": qa.get("query", ""),
                    "answer": qa.get("answer", ""),
                    "ground_truth_chunk_key": _doc_key(text, meta),
                    "subject": subject,
                    "source": meta.get("source", ""),
                    "chunk_preview": text[:200].replace("\n", " "),
                })
                total_generated += 1

    output_path = TESTSET_DIR / output_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"测试集生成完成！共 {total_generated} 条 QA pairs")
    print(f"保存至: {output_path}")
    print(f"{'='*50}")
    return output_path


if __name__ == "__main__":
    import sys
    # 允许命令行参数：python generate_testset.py --samples 20 --questions 2
    samples = 15
    questions = 1
    if "--samples" in sys.argv:
        idx = sys.argv.index("--samples")
        samples = int(sys.argv[idx + 1])
    if "--questions" in sys.argv:
        idx = sys.argv.index("--questions")
        questions = int(sys.argv[idx + 1])

    build_testset(samples_per_subject=samples, questions_per_chunk=questions)
