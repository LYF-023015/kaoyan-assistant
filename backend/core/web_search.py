"""
联网搜索工具
默认使用 DuckDuckGo，预留 Bing/Google API 扩展
"""
from typing import List, Dict


class WebSearchTool:
    def __init__(self, provider: str = "duckduckgo"):
        self.provider = provider

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        if self.provider == "duckduckgo":
            return self._search_duckduckgo(query, max_results)
        else:
            return []

    def _search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                return [
                    {
                        "title": r.get("title", ""),
                        "href": r.get("href", ""),
                        "body": r.get("body", ""),
                    }
                    for r in results
                ]
        except Exception as e:
            print(f"DuckDuckGo 搜索失败: {e}")
            return []


def web_search(query: str, max_results: int = 5) -> str:
    """
    搜索并返回格式化的文本结果
    """
    tool = WebSearchTool()
    results = tool.search(query, max_results)
    if not results:
        return "未找到相关网络搜索结果。"

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['title']}\n{r['body']}\n来源: {r['href']}")
    return "\n\n".join(lines)
