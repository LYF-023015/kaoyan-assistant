"""
RAG 评估指标模块
包含两类指标：
1. 检索指标（需要 ground truth chunk keys）: Recall@K, Precision@K, HitRate@K, MRR
2. 生成指标（LLM-as-a-Judge）: Faithfulness, Answer Relevance, Context Precision
"""
import json
from typing import List, Dict, Tuple
from backend.core.document_processor import Document
from backend.llm import get_llm_client


# ==================== 检索指标 ====================

def _rank_of_first_relevant(retrieved_keys: List[str], relevant_keys: set) -> int:
    """返回第一个 relevant chunk 的 rank（1-based），未找到返回 0"""
    for rank, key in enumerate(retrieved_keys, start=1):
        if key in relevant_keys:
            return rank
    return 0


def recall_at_k(retrieved_keys: List[str], relevant_keys: set, k: int) -> float:
    """
    Recall@K = |relevant ∩ retrieved[:K]| / |relevant|
    """
    if not relevant_keys:
        return 0.0
    retrieved_set = set(retrieved_keys[:k])
    hits = len(relevant_keys & retrieved_set)
    return hits / len(relevant_keys)


def precision_at_k(retrieved_keys: List[str], relevant_keys: set, k: int) -> float:
    """
    Precision@K = |relevant ∩ retrieved[:K]| / K
    """
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_keys[:k])
    hits = len(relevant_keys & retrieved_set)
    return hits / k


def hit_rate_at_k(retrieved_keys: List[str], relevant_keys: set, k: int) -> float:
    """
    Hit Rate@K = 1 if any relevant in top-K else 0
    """
    retrieved_set = set(retrieved_keys[:k])
    return 1.0 if (relevant_keys & retrieved_set) else 0.0


def mrr(retrieved_keys: List[str], relevant_keys: set) -> float:
    """
    Mean Reciprocal Rank = 1 / rank_of_first_relevant
    """
    rank = _rank_of_first_relevant(retrieved_keys, relevant_keys)
    return 1.0 / rank if rank > 0 else 0.0


def compute_retrieval_metrics(
    retrieved_keys: List[str],
    relevant_keys: set,
    k_values: List[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """计算一组检索指标"""
    results = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved_keys, relevant_keys, k)
        results[f"precision@{k}"] = precision_at_k(retrieved_keys, relevant_keys, k)
        results[f"hit_rate@{k}"] = hit_rate_at_k(retrieved_keys, relevant_keys, k)
    results["mrr"] = mrr(retrieved_keys, relevant_keys)
    return results


# ==================== LLM-as-a-Judge 指标 ====================

class LLMJudge:
    """使用 LLM 评估生成质量"""

    def __init__(self):
        self.llm = get_llm_client()

    def _call_llm(self, prompt: str, temperature: float = 0.1) -> str:
        try:
            return self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
            )
        except Exception as e:
            print(f"  LLM Judge 调用失败: {e}")
            return ""

    def faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Faithfulness (忠实度): 答案中的 claims 有多少能从上下文中推断出来。
        返回 0.0 ~ 1.0 的分数。
        """
        context_text = "\n\n---\n\n".join(contexts)
        prompt = f"""你是一个严格的的事实核查员。请评估以下答案是否忠实于给定的上下文。

【上下文】
{context_text[:3000]}

【答案】
{answer[:1500]}

任务：
1. 从答案中提取所有事实性陈述（claims）
2. 逐一检查每个陈述是否能从上下文中直接推断或明确支持
3. 统计：被支持的陈述数 / 总陈述数

请只输出一个 0.0 到 1.0 之间的数字，表示忠实度分数。不要输出任何解释。
输出格式示例：0.85"""

        response = self._call_llm(prompt).strip()
        # 尝试提取数字
        try:
            # 找最后一个看起来像数字的
            for token in response.replace(",", "").split():
                try:
                    score = float(token)
                    if 0.0 <= score <= 1.0:
                        return round(score, 3)
                except ValueError:
                    continue
            # 如果都不行，尝试整个字符串
            score = float(response)
            return round(max(0.0, min(1.0, score)), 3)
        except Exception:
            return 0.0

    def answer_relevance(self, query: str, answer: str) -> float:
        """
        Answer Relevance (回答相关性): 答案是否直接、完整地回答了问题。
        返回 0.0 ~ 1.0 的分数。
        """
        prompt = f"""你是一个阅卷专家。请评估以下答案对问题的相关性和完整性。

【问题】
{query}

【答案】
{answer[:1500]}

评分标准（0.0 ~ 1.0）：
- 1.0: 答案直接、完整、准确地回答了问题
- 0.7: 答案基本回答了问题，但略有遗漏或不精确
- 0.4: 答案部分相关，但偏离了核心问题或信息不足
- 0.1: 答案与问题几乎无关

请只输出一个 0.0 到 1.0 之间的数字。不要输出任何解释。"""

        response = self._call_llm(prompt).strip()
        try:
            for token in response.replace(",", "").split():
                try:
                    score = float(token)
                    if 0.0 <= score <= 1.0:
                        return round(score, 3)
                except ValueError:
                    continue
            score = float(response)
            return round(max(0.0, min(1.0, score)), 3)
        except Exception:
            return 0.0

    def context_precision(self, query: str, contexts: List[str]) -> float:
        """
        Context Precision (上下文精确率): 检索到的 chunks 中有多少是相关的。
        返回 0.0 ~ 1.0 的分数。
        """
        if not contexts:
            return 0.0

        # 逐条评估每个 context 是否相关
        relevant_count = 0
        for ctx in contexts:
            prompt = f"""判断以下检索到的文本片段是否与用户问题相关。

【问题】{query}

【文本片段】
{ctx[:800]}

该片段是否包含回答问题所需的信息？请只回答 "是" 或 "否"。"""
            response = self._call_llm(prompt).strip().lower()
            if "是" in response or "yes" in response:
                relevant_count += 1

        return round(relevant_count / len(contexts), 3)


# ==================== 汇总报告 ====================

def print_report(retrieval_metrics: List[Dict], generation_metrics: List[Dict], k_values: List[int] = (1, 3, 5, 10)):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("RAG 评估报告")
    print("=" * 60)

    n = len(retrieval_metrics)
    if n == 0:
        print("无评估数据")
        return

    # 检索指标平均值
    print(f"\n【检索指标】样本数: {n}")
    for k in k_values:
        avg_recall = sum(m[f"recall@{k}"] for m in retrieval_metrics) / n
        avg_precision = sum(m[f"precision@{k}"] for m in retrieval_metrics) / n
        avg_hit = sum(m[f"hit_rate@{k}"] for m in retrieval_metrics) / n
        print(f"  Recall@{k:<3}  = {avg_recall:.3f}  |  Precision@{k:<3} = {avg_precision:.3f}  |  HitRate@{k:<3} = {avg_hit:.3f}")

    avg_mrr = sum(m["mrr"] for m in retrieval_metrics) / n
    print(f"  MRR        = {avg_mrr:.3f}")

    # 生成指标平均值
    if generation_metrics:
        n_gen = len(generation_metrics)
        print(f"\n【生成指标】样本数: {n_gen}")
        for metric_name in ["faithfulness", "answer_relevance", "context_precision"]:
            values = [m.get(metric_name, 0.0) for m in generation_metrics if metric_name in m]
            if values:
                avg = sum(values) / len(values)
                print(f"  {metric_name:<20} = {avg:.3f}")

    print("=" * 60)
