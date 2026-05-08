"""
考研助手全局配置
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 路径配置
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
BM25_INDEX_PATH = VECTOR_DB_DIR / "bm25_index.pkl"
CHROMA_PERSIST_DIR = VECTOR_DB_DIR / "chroma"

# 模型配置
M3E_MODEL_PATH = os.getenv("M3E_MODEL_PATH", r"E:\models\m3e-base")
M3E_MODEL_HF = "moka-ai/m3e-base"

BGE_RERANKER_PATH = os.getenv("BGE_RERANKER_PATH", "BAAI/bge-reranker-base")

DEVICE = os.getenv("DEVICE", "auto")  # auto / cuda / cpu

# LLM 配置
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "zhipuai")  # openai / zhipuai / anthropic / mimo / kimi
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# 小米 MiMo
MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")

# Kimi 2.6 Code Plan
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.kimi.com/coding/")

# MinerU 云端 API
MINERU_API_URL = os.getenv("MINERU_API_URL", "")
MINERU_API_KEY = os.getenv("MINERU_API_KEY", "")

# 学科列表
SUBJECTS = ["政治", "英语", "数学", "自动控制原理"]

# 学科名称映射（用于 ChromaDB collection name，只能含 ASCII）
SUBJECT_NAME_MAP = {
    "政治": "politics",
    "英语": "english",
    "数学": "math",
    "自动控制原理": "control",
}

# 检索配置
DENSE_TOP_K = 50
SPARSE_TOP_K = 50
RRF_K = 60
RERANK_TOP_K = 20
FINAL_TOP_K = 10

# ReAct Agent
MAX_AGENT_ITERATIONS = 3

# 文本分块
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# 确保目录存在
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
for subj in SUBJECTS:
    (KNOWLEDGE_BASE_DIR / subj).mkdir(parents=True, exist_ok=True)
