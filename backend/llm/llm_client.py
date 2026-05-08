"""
统一 LLM Client，兼容 OpenAI / Anthropic / Zhipu / MiMo / Kimi 格式
"""
import os
from typing import List, Dict, Optional, Iterator
from openai import OpenAI
import anthropic


class LLMClient:
    def __init__(
        self,
        provider: str = "zhipuai",
        model: str = "glm-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 自动补全 base_url 和 api_key
        if self.provider == "zhipuai":
            self.base_url = base_url or "https://open.bigmodel.cn/api/paas/v4/"
            self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY", "")
        elif self.provider == "openai":
            self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            self.base_url = None
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        elif self.provider == "mimo":
            self.base_url = base_url or os.getenv("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
            self.api_key = api_key or os.getenv("MIMO_API_KEY", "")
        elif self.provider == "kimi":
            # Kimi 2.6 Code Plan 兼容 Anthropic 格式
            self.base_url = base_url or os.getenv("KIMI_BASE_URL", "https://api.kimi.com/coding/")
            self.api_key = api_key or os.getenv("KIMI_API_KEY", "")
        else:
            self.base_url = base_url or ""
            self.api_key = api_key or ""

        self._openai_client = None
        self._anthropic_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._openai_client

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            kwargs = {"api_key": self.api_key}
            # Kimi 等第三方兼容服务需要自定义 base_url
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._anthropic_client = anthropic.Anthropic(**kwargs)
        return self._anthropic_client

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        非流式对话
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider in ("anthropic", "kimi"):
            return self._chat_anthropic(messages, stream=False, temperature=temp, max_tokens=max_tok)
        else:
            return self._chat_openai_format(messages, stream=False, temperature=temp, max_tokens=max_tok)

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        """
        流式对话，返回 token 生成器
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider in ("anthropic", "kimi"):
            yield from self._chat_anthropic_stream(messages, temperature=temp, max_tokens=max_tok)
        else:
            yield from self._chat_openai_format_stream(messages, temperature=temp, max_tokens=max_tok)

    def _chat_openai_format(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            return ""
        return response.choices[0].message.content or ""

    def _chat_openai_format_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        stream: bool,
        temperature: float,
        max_tokens: int,
    ) -> str:
        client = self._get_anthropic_client()
        system_msg = ""
        anthropic_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                anthropic_messages.append({"role": m["role"], "content": m["content"]})

        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=anthropic_messages,
            stream=stream,
        )
        if stream:
            return ""
        return response.content[0].text if response.content else ""

    def _chat_anthropic_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        client = self._get_anthropic_client()
        system_msg = ""
        anthropic_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                anthropic_messages.append({"role": m["role"], "content": m["content"]})

        with client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=anthropic_messages,
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text


# 全局单例
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        from backend.config import LLM_PROVIDER, LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS
        _llm_client = LLMClient(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _llm_client
