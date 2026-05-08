"""
RAG 系统主评估脚本

使用方式:
  1. 先生成测试集:
     python backend/eval/generate_testset.py

  2. 运行评估:
     python backend/eval/evaluate.py --testset backend/eval/testset/testset.json

  3. 只评估检索（速度快，不需要 LLM 生成答案）:
     python backend/eval/evaluate.py --testset ... --retrieval-only

  4. 评估生成质量（需要 LLM 调用，较慢）:
     python backend/eval/evaluate.py --testset ... --generation-only --sample 20
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# 确保项目根目录在路径中
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.core.retriever import FusionRetriever, get_retriever
from backend.core.react_agent import get_agent
from backend.eval.metrics import (
    compute_retrieval_metrics,
    LLMJudge,
    print_report,
)


def load_testset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 过滤掉空 query
    return [item for item in data if item.get("query", "").strip()]


def evaluate_retrieval(test_cases: List[Dict], retriever: FusionRetriever) -> List[Dict]:
    """评估检索质量，返回每条 case 的指标"""
    results = []
    print(f"\n开始评估检索质量，共 {len(test_cases)} 条...")

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        subject = case.get("subject", "全部")
        gt_key = case.get("ground_truth_chunk_key", "")
        relevant_keys = {gt_key} if gt_key else set()

        # 执行检索
        retrieved = retriever.retrieve(query, subject=subject, use_reranker=True)
        retrieved_keys = []
        for doc, score in retrieved:
            meta = doc.metadata
            key = f"{meta.get('source', '')}_{meta.get('chunk_index', 0)}_{hash(doc.content) & 0xFFFFFF}"
            retrieved_keys.append(key)

        # 计算指标
        metrics = compute_retrieval_metrics(retrieved_keys, relevant_keys, k_values=[1, 3, 5, 10])
        results.append(metrics)

        if i % 10 == 0 or i == len(test_cases):
            print(f"  已处理 {i}/{len(test_cases)}  |  当前 Recall@10={metrics['recall@10']:.2f}  MRR={metrics['mrr']:.3f}")

    return results


def evaluate_generation(
    test_cases: List[Dict],
    retriever: FusionRetriever,
    judge: LLMJudge,
    sample_size: int = None,
) -> List[Dict]:
    """评估生成质量，返回每条 case 的指标"""
    if sample_size and sample_size < len(test_cases):
        import random
        cases = random.sample(test_cases, sample_size)
        print(f"\n开始评估生成质量，从 {len(test_cases)} 条中采样 {len(cases)} 条...")
    else:
        cases = test_cases
        print(f"\n开始评估生成质量，共 {len(cases)} 条...")

    results = []
    agent = get_agent()

    for i, case in enumerate(cases, 1):
        query = case["query"]
        subject = case.get("subject", "全部")

        # 检索上下文
        retrieved = retriever.retrieve(query, subject=subject, use_reranker=True)
        contexts = [doc.content for doc, _ in retrieved]

        # 生成答案（通过 agent 或直接 LLM）
        # 这里用 agent.run 模拟真实场景
        try:
            answer, _ = agent.run(query=query, subject=subject, use_web_search=False, history=[])
        except Exception as e:
            print(f"  生成答案失败: {e}")
            answer = ""

        # LLM 评估
        metrics = {}
        if answer:
            metrics["faithfulness"] = judge.faithfulness(answer, contexts)
            metrics["answer_relevance"] = judge.answer_relevance(query, answer)
            metrics["context_precision"] = judge.context_precision(query, contexts)
        else:
            metrics["faithfulness"] = 0.0
            metrics["answer_relevance"] = 0.0
            metrics["context_precision"] = 0.0

        results.append(metrics)
        print(f"  [{i}/{len(cases)}] Faithfulness={metrics['faithfulness']:.2f}  "
              f"Relevance={metrics['answer_relevance']:.2f}  "
              f"CtxPrecision={metrics['context_precision']:.2f}")

    return results


def evaluate_ablation(
    test_cases: List[Dict],
    retriever: FusionRetriever,
) -> None:
    """
    消融实验：对比不同检索配置的指标差异
    - 只用 Dense (M3E)
    - 只用 Sparse (BM25)
    - Dense + Sparse (RRF)
    - Dense + Sparse + Reranker (完整)
    """
    print("\n" + "=" * 60)
    print("消融实验：不同检索配置对比")
    print("=" * 60)

    configs = [
        ("Dense only (M3E)", {"use_bm25": False, "use_rrf": False, "use_reranker": False}),
        ("Sparse only (BM25)", {"use_bm25": True, "use_dense": False, "use_rrf": False, "use_reranker": False}),
        ("RRF (Dense + Sparse)", {"use_bm25": True, "use_dense": True, "use_rrf": True, "use_reranker": False}),
        ("Full (RRF + Reranker)", {"use_bm25": True, "use_dense": True, "use_rrf": True, "use_reranker": True}),
    ]

    # 需要临时修改 retriever 的行为... 这个比较复杂
    # 简单起见，先打印提示
    print("\n[提示] 消融实验需要修改 retriever 接口以支持开关各组件。")
    print("当前 retriever 内部耦合较紧，建议先评估完整 pipeline 的指标。")


def main():
    parser = argparse.ArgumentParser(description="RAG 系统评估")
    parser.add_argument("--testset", type=str, required=True, help="测试集 JSON 文件路径")
    parser.add_argument("--retrieval-only", action="store_true", help="只评估检索指标")
    parser.add_argument("--generation-only", action="store_true", help="只评估生成指标")
    parser.add_argument("--sample", type=int, default=None, help="生成评估采样数量（降低 LLM 调用成本）")
    parser.add_argument("--output", type=str, default=None, help="结果保存路径")
    args = parser.parse_args()

    # 加载测试集
    test_cases = load_testset(args.testset)
    print(f"加载测试集: {args.testset}")
    print(f"共 {len(test_cases)} 条测试用例")

    if not test_cases:
        print("测试集为空，请先运行 generate_testset.py 生成测试集")
        return

    retriever = get_retriever()
    retrieval_results = []
    generation_results = []

    # 检索评估
    if not args.generation_only:
        retrieval_results = evaluate_retrieval(test_cases, retriever)

    # 生成评估
    if not args.retrieval_only:
        judge = LLMJudge()
        generation_results = evaluate_generation(
            test_cases, retriever, judge, sample_size=args.sample
        )

    # 打印报告
    print_report(retrieval_results, generation_results)

    # 保存详细结果
    if args.output:
        output = {
            "testset": args.testset,
            "num_cases": len(test_cases),
            "retrieval_metrics": retrieval_results,
            "generation_metrics": generation_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
