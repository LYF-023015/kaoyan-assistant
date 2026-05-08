"""
ReAct Agent 实现
工具：
  - retrieve_knowledge(query, subject): 检索本地知识库
  - web_search(query): 联网搜索
"""
import json
import re
from typing import List, Dict, Optional, Tuple
from backend.llm import get_llm_client
from backend.core.retriever import get_retriever
from backend.core.web_search import web_search
from backend.config import MAX_AGENT_ITERATIONS


REACT_SYSTEM_PROMPT = """你是一个专业的考研辅导助手，擅长使用工具帮助用户解答考研相关问题。

你可以使用以下工具：
1. retrieve_knowledge(query, subject) - 从本地考研知识库中检索相关资料
2. web_search(query) - 在互联网上搜索最新信息

你的思考过程必须严格按照以下 ReAct 格式输出：
Thought: 你的思考过程
Action: {{"tool": "工具名", "args": {{"参数名": "参数值"}}}}
Observation: 工具返回的结果（由系统自动填充，你不需要输出）

你可以进行最多 {max_iterations} 轮工具调用。当你认为已经获得足够信息时，输出：
Thought: 我已经获得足够信息，可以回答用户问题了。
Action: {{"tool": "FinalAnswer", "args": {{"answer": "你的最终答案"}}}}

注意事项：
- 回答要准确、有条理，优先引用本地知识库的内容
- 如果涉及数学公式，使用 LaTeX 格式
- 分点阐述，保持简明扼要
- 如果信息不足，明确告知用户
"""


class ReActAgent:
    def __init__(self):
        self.llm = get_llm_client()
        self.retriever = get_retriever()
        self.max_iterations = MAX_AGENT_ITERATIONS

    def run(
        self,
        query: str,
        subject: Optional[str] = None,
        use_web_search: bool = True,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        执行 ReAct 循环
        返回: (final_answer, retrieved_sources)
        """
        sources = []
        messages = []
        system_prompt = REACT_SYSTEM_PROMPT.format(max_iterations=self.max_iterations)
        messages.append({"role": "system", "content": system_prompt})

        # 构建初始用户消息
        user_msg = f"用户问题: {query}\n"
        if subject and subject != "全部":
            user_msg += f"学科: {subject}\n"
        user_msg += "请使用工具获取信息并回答。"
        messages.append({"role": "user", "content": user_msg})

        try:
            for step in range(self.max_iterations):
                # 调用 LLM
                response = self.llm.chat(messages, stream=False)

                # 解析 Thought 和 Action
                thought, action_json = self._parse_response(response)

                if not action_json:
                    # 无法解析，直接当作最终答案
                    return response, sources

                tool_name = action_json.get("tool", "")
                args = action_json.get("args", {})

                if tool_name == "FinalAnswer":
                    answer = args.get("answer", response)
                    return answer, sources

                # 执行工具
                observation = self._execute_tool(tool_name, args, subject, sources, use_web_search)

                # 将 Assistant 输出和 Observation 加入上下文
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            # 达到最大迭代次数，强制总结
            messages.append({"role": "user", "content": "已达到最大工具调用次数，请基于已有信息给出最终答案。"})
            final_response = self.llm.chat(messages, stream=False)
            return final_response, sources
        except Exception as e:
            error_msg = f"LLM 调用失败: {str(e)}。请检查 API Key 配置是否正确。"
            return error_msg, sources

    def _parse_response(self, response: str) -> Tuple[str, Optional[Dict]]:
        """
        解析 LLM 的 ReAct 输出
        """
        thought_match = re.search(r'Thought:\s*(.+?)(?:\nAction:|$)', response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

        action_match = re.search(r'Action:\s*(\{.*?\})', response, re.DOTALL)
        if not action_match:
            return thought, None

        try:
            action_json = json.loads(action_match.group(1))
            return thought, action_json
        except json.JSONDecodeError:
            # 尝试更宽松的提取
            try:
                # 寻找第一个 { 和最后一个 }
                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    action_json = json.loads(response[start:end+1])
                    return thought, action_json
            except Exception:
                pass
            return thought, None

    def _execute_tool(
        self,
        tool_name: str,
        args: Dict,
        subject: Optional[str],
        sources: List[Dict],
        use_web_search: bool,
    ) -> str:
        """
        执行工具调用
        """
        if tool_name == "retrieve_knowledge":
            q = args.get("query", "")
            subj = args.get("subject", subject)
            results = self.retriever.retrieve(q, subject=subj, use_reranker=True)
            if not results:
                return "未在本地知识库中找到相关资料。"

            lines = []
            for doc, score in results:
                meta = doc.metadata
                lines.append(
                    f"[来源: {meta.get('source', '未知')}, 分数: {score:.4f}]\n{doc.content[:500]}"
                )
                sources.append({
                    "content": doc.content,
                    "metadata": meta,
                    "score": score,
                })
            return "\n\n".join(lines)

        elif tool_name == "web_search" and use_web_search:
            q = args.get("query", "")
            return web_search(q, max_results=5)

        else:
            return f"未知工具: {tool_name}"


# 全局单例
_agent = None


def get_agent() -> ReActAgent:
    global _agent
    if _agent is None:
        _agent = ReActAgent()
    return _agent
