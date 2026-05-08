# 考研助手 RAG 系统

基于 **M3E 稠密召回 + BM25 稀疏召回 + RRF 融合 + BGE-Reranker 重排** 的考研知识库问答系统，采用 **ReAct Agent** 架构，支持联网搜索增强，前端为 Kimi 风格流式交互界面。

---

## 目录

- [系统架构](#系统架构)
- [算法原理](#算法原理)
  - [双路召回](#双路召回)
  - [RRF 融合](#rrf-融合)
  - [BGE-Reranker 重排序](#bge-reranker-重排序)
  - [ReAct Agent](#react-agent)
  - [增量索引机制](#增量索引机制)
  - [PDF 解析策略](#pdf-解析策略)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置详解](#配置详解)
- [RAG 评估](#rag-评估)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              用户层 (Frontend)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  学科选择     │  │  文件上传     │  │  聊天输入     │  │  联网开关     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
│                              SSE 流式输出                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            API 层 (FastAPI)                              │
│  POST /api/upload        →  保存文件 + 自动触发增量索引                   │
│  POST /api/index         →  SSE 流式索引构建/更新                        │
│  POST /api/chat/stream   →  SSE 流式对话                                │
│  GET  /api/files         →  知识库文件列表                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           ReAct Agent 决策层                              │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  System: 你是考研辅导专家。可用工具：retrieve_knowledge, web_search ││
│  │                                                                     ││
│  │  Thought 1: 用户问的是数学问题，先查本地知识库                      ││
│  │  Action 1: retrieve_knowledge(query="求极限方法", subject="数学")   ││
│  │  Observation 1: [检索到 10 个相关 chunk]                            ││
│  │                                                                     ││
│  │  Thought 2: 本地资料足够，无需联网                                  ││
│  │  Action 2: final_answer(答案...)                                    ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         检索层 (Fusion Retriever)                        │
│                                                                         │
│   用户查询 ──►┌─────────────────┐    ┌─────────────────┐                │
│               │  Dense Retrieval│    │ Sparse Retrieval│                │
│               │  M3E Embedding  │    │  BM25 + jieba   │                │
│               │  ChromaDB cosine│    │  rank_bm25      │                │
│               │  top-k = 50     │    │  top-k = 50     │                │
│               └────────┬────────┘    └────────┬────────┘                │
│                        │                      │                         │
│                        └──────────┬───────────┘                         │
│                                   ▼                                     │
│                          ┌─────────────────┐                            │
│                          │  RRF Fusion     │                            │
│                          │  k = 60         │                            │
│                          └────────┬────────┘                            │
│                                   ▼                                     │
│                          ┌─────────────────┐                            │
│                          │ BGE-Reranker    │                            │
│                          │ top-20 → top-10 │                            │
│                          └────────┬────────┘                            │
│                                   ▼                                     │
│                          返回 (Document, score) × 10                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          存储层                                          │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐ │
│  │ ChromaDB (持久化)    │  │ BM25 索引 (pickle)   │  │ index_state.json │ │
│  │ exam_math           │  │ bm25_index.pkl        │  │ 文件状态追踪      │ │
│  │ exam_english        │  │                       │  │                  │ │
│  │ exam_politics       │  │                       │  │                  │ │
│  │ exam_control        │  │                       │  │                  │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────┘ │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ knowledge_base/                                                      ││
│  │ ├── 数学/ (PDF 教材)                                                  ││
│  │ ├── 英语/                                                             ││
│  │ ├── 政治/                                                             ││
│  │ └── 自动控制原理/                                                      ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 算法原理

### 双路召回

系统同时运行两条独立的召回路径，弥补单一方法的缺陷：

#### 稠密召回 (Dense Retrieval)

**模型**: M3E-base (moka-ai/m3e-base)，768 维向量

**原理**: 将查询和文档编码为语义向量，通过余弦相似度衡量语义相关性。

```
query_embedding = M3E(query)          # 768-dim
chunk_embedding = M3E(chunk_text)     # 768-dim
similarity = cosine(query_embedding, chunk_embedding)
```

**优点**: 能理解同义词、语义相近表达（如"求导"与"微分"）
**缺点**: 对精确术语匹配较弱；短查询 embedding 噪声大

#### 稀疏召回 (Sparse Retrieval)

**模型**: BM25Okapi + jieba 中文分词

**原理**: 基于词频统计的经典信息检索算法，对精确关键词匹配效果好。

```
BM25(q, d) = Σ idf(qi) · [ f(qi,d) · (k1+1) ] / [ f(qi,d) + k1 · (1-b+b·|d|/avgdl) ]

其中:
  f(qi,d)  = 词 qi 在文档 d 中的词频
  |d|      = 文档长度
  avgdl    = 平均文档长度
  k1 = 1.5, b = 0.75 (默认参数)
  idf(qi)  = log( (N - n(qi) + 0.5) / (n(qi) + 0.5) )
```

**jieba 分词策略**: 使用 `cut_for_search` 模式，对查询和文档都做细粒度切分，提高召回率。

**优点**: 精确术语匹配强（如公式名称、定理名）
**缺点**: 无法理解语义相似但字面不同的表达

### RRF 融合

**Reciprocal Rank Fusion** 将两条召回路径的结果合并为一个统一排名。

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
               i∈{dense, sparse}

其中:
  rank_i(d) = 文档 d 在第 i 个列表中的排名 (1-based)
  k = 60 (平滑常数，防止低排名文档分数过高)
```

**为什么用 RRF 而不是加权求和？**
- RRF 不需要校准两个不同量纲的分数（余弦相似度 vs BM25 分数）
- 对排名位置敏感，但对绝对分数不敏感，更鲁棒
- 已被大量论文验证有效（Cormack et al., 2009）

**示例**:

| 文档 | Dense 排名 | Sparse 排名 | RRF 分数 (k=60) |
|------|-----------|------------|----------------|
| D1   | 1         | 5          | 1/61 + 1/65 = 0.0164 + 0.0154 = **0.0318** |
| D2   | 3         | 2          | 1/63 + 1/62 = 0.0159 + 0.0161 = **0.0320** |
| D3   | 2         | 未命中     | 1/62 + 0      = **0.0161** |

D2 在两条路径上都表现不错，通过 RRF 排在首位。

### BGE-Reranker 重排序

RRF 融合后取前 20 名，送入 BGE-Reranker 做更精细的语义相关性判断。

**模型**: BAAI/bge-reranker-base

**原理**: 将 `[查询, 文档]` 拼接为句子对，输入 Cross-Encoder，输出相关性概率。

```
输入:  [ [query, doc1], [query, doc2], ... ]
      ↓
BERT Cross-Encoder
      ↓
输出 logits: [ [neg_score, pos_score], ... ]
      ↓
softmax → 取正类概率作为 relevance_score
      ↓
按 relevance_score 降序排列，取 TOP-10
```

**为什么需要 Reranker？**
- 召回阶段追求"不漏"（高 Recall）
- Reranker 追求"精准"（高 Precision），用更强的模型做精排
- Cross-Encoder 比 Bi-Encoder（M3E）有更强的交互能力，因为 query 和 doc 在 attention 层直接交互

### ReAct Agent

**ReAct = Reasoning + Acting**，让 LLM 在生成答案前先思考需要什么信息，再主动调用工具获取。

```
系统 Prompt 给 LLM 的格式:

你可以使用以下工具:
1. retrieve_knowledge(query, subject) - 从本地知识库检索
2. web_search(query) - 联网搜索

思考步骤:
Thought: 用户问的是... 我应该先...
Action: retrieve_knowledge(query="...", subject="数学")
Observation: [检索结果...]

Thought: 根据检索结果，我还需要...
Action: web_search(query="...")
Observation: [搜索结果...]

Thought: 信息已足够，给出最终答案
Action: final_answer(答案...)
```

**决策逻辑**:
1. 第一轮：先调用 `retrieve_knowledge`，获取本地资料
2. 如果检索结果不足（如返回 chunk 数 < 3），自动调用 `web_search` 补充
3. 最多 3 轮迭代，防止无限循环
4. 最终用检索到的上下文作为 system prompt，LLM 生成带引用来源的答案

### 增量索引机制

传统 RAG 系统每次"重建索引"都要解析所有文件、重新生成所有向量，对于上百个 PDF 非常耗时。

本系统采用**增量索引**策略：

```
文件系统              index_state.json              ChromaDB
   │                        │                          │
   │  上传新文件              │                          │
   ▼                        ▼                          ▼
knowledge_base/        记录每个文件的:           已索引的文档
├── 数学/              - mtime (修改时间)        exam_math
│   ├── a.pdf          - size (文件大小)           ├── doc_uuid_1
│   └── b.pdf   ─────► - doc_ids (向量库ID列表)    ├── doc_uuid_2
└── 英语/              - chunks (文本块数量)       └── ...
    └── c.pdf
```

**索引流程**:
1. 扫描 `knowledge_base/{subject}/` 目录
2. 对比 `index_state.json`：
   - `mtime` + `size` 都一致 → **跳过**（已有索引，无需处理）
   - 文件不在 state 中，或 `mtime/size` 变化 → **重新解析并添加**
   - state 中有但目录中不存在 → **从 ChromaDB 删除对应 doc_ids**
3. 只处理新增/修改的文件，已有文件绝不清空
4. 更新 BM25 索引（BM25 不支持增量，从 ChromaDB 全量拉取重建，但很快）

**状态文件丢失的容错**: 如果 `index_state.json` 被删除，系统会从 ChromaDB 的 metadata 中自动重建 state，不会导致重复索引。

### PDF 解析策略

三层降级策略，确保最大程度提取结构化内容：

```
PDF 文件
    │
    ├──► MinerU 云端 API（首选）
    │      流程: 申请上传URL → PUT上传 → 轮询解析(最多5min)
    │             → 下载ZIP → 解压提取 Markdown
    │      优点: 公式识别强、保留标题层级、图片有描述
    │      缺点: 依赖网络、有文件大小限制(200MB)
    │
    ├──► MinerU 本地版 magic_pdf（备选）
    │      优点: 无需网络、隐私性好
    │      缺点: 需要下载大模型(~10GB)
    │
    └──► PyMuPDF（降级兜底）
           纯文本提取，无结构化信息
           公式会变成乱码或缺失
```

**Markdown 分块策略**: MinerU 输出的 Markdown 按 `##` / `###` 标题分块，保持章节完整性。超过 2000 字符的大章节会按段落二次切分。

---

## 项目结构

```
kaoyan-assistant/
├── backend/
│   ├── main.py                     # FastAPI 入口
│   ├── api/
│   │   └── routes.py               # API 路由 + SSE 流式索引
│   ├── core/
│   │   ├── pdf_parser.py           # PDF 解析（MinerU + PyMuPDF）
│   │   ├── document_processor.py   # 文档加载、分块、元数据注入
│   │   ├── embeddings.py           # M3E Embedding 模型封装
│   │   ├── vector_store.py         # ChromaDB 持久化向量库
│   │   ├── bm25_index.py           # BM25 稀疏索引
│   │   ├── retriever.py            # 双路召回 + RRF + BGE-Reranker
│   │   ├── react_agent.py          # ReAct Agent 决策循环
│   │   └── web_search.py           # DuckDuckGo 联网搜索
│   ├── eval/                       # RAG 评估框架
│   │   ├── generate_testset.py     # LLM 自动生成测试 QA pairs
│   │   ├── metrics.py              # 检索指标 + LLM-as-Judge
│   │   └── evaluate.py             # 主评估脚本
│   ├── llm/
│   │   └── llm_client.py           # 统一 LLM Client（OpenAI/Anthropic/Kimi/GLM/MiMo）
│   └── config.py                   # 全局配置
│
├── frontend/
│   ├── index.html                  # 单页面应用
│   ├── css/style.css               # Kimi 风格暗色主题
│   └── js/
│       ├── app.js                  # 主控制器（上传、索引、聊天）
│       ├── chat.js                 # Markdown + LaTeX 渲染
│       └── api.js                  # API 调用封装
│
├── knowledge_base/                 # 本地知识库（按学科分类）
│   ├── 政治/
│   ├── 英语/
│   ├── 数学/
│   └── 自动控制原理/
│
├── vector_db/                      # 持久化数据
│   ├── chroma/                     # ChromaDB 向量数据
│   ├── bm25_index.pkl              # BM25 索引文件
│   └── index_state.json            # 增量索引状态追踪
│
├── start.py                        # 一键启动脚本
├── requirements.txt                # Python 依赖
└── .env                            # API Key 与环境配置
```

---

## 快速开始

### 1. 环境准备

```bash
# 使用 conda 环境（推荐）
conda activate llm-universe

# 或创建新环境
conda create -n kaoyan python=3.10
conda activate kaoyan
pip install -r requirements.txt
```

> **注意**: `torch`、`transformers`、`sentence-transformers` 体积较大。网络不佳时使用清华镜像：
> ```bash
> pip install torch transformers sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 2. 配置 API Key

复制或编辑项目根目录的 `.env` 文件：

```ini
# ========== LLM 配置（必填其一） ==========
LLM_PROVIDER=kimi
LLM_MODEL=kimi-k2.5-coding
LLM_API_KEY=sk-your-key

# ========== MinerU 云端解析（可选） ==========
# 配置了则优先使用云端，否则尝试本地 MinerU → PyMuPDF
MINERU_API_URL=https://mineru.net/api/v4
MINERU_API_KEY=eyJ...

# ========== 本地模型路径（可选） ==========
M3E_MODEL_PATH=E:\models\m3e-base
BGE_RERANKER_PATH=BAAI/bge-reranker-base

# ========== 设备 ==========
DEVICE=auto   # auto / cuda / cpu
```

支持的 LLM 提供商: `kimi` | `zhipuai` | `openai` | `anthropic` | `mimo`

### 3. 启动服务

```bash
python start.py
# 或指定端口
python start.py --port 8000
```

访问:
- **前端界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

### 4. 上传资料

1. 在左侧边栏选择学科
2. 点击 **上传资料** 按钮，选择 PDF/TXT/MD/DOCX
3. 文件保存后**自动触发增量索引**，进度实时显示在聊天区
4. 已有文件不会被重复处理

### 5. 开始提问

- 输入框支持 `Shift + Enter` 换行
- 可切换是否启用联网搜索
- 点击参考来源可展开查看原始文档片段

---

## 配置详解

### 检索参数 (`backend/config.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DENSE_TOP_K` | 50 | M3E 稠密召回数量 |
| `SPARSE_TOP_K` | 50 | BM25 稀疏召回数量 |
| `RRF_K` | 60 | RRF 平滑常数 k |
| `RERANK_TOP_K` | 20 | 送入 Reranker 的文档数 |
| `FINAL_TOP_K` | 10 | 最终返回的文档数 |
| `CHUNK_SIZE` | 800 | 文本分块目标长度 |
| `CHUNK_OVERLAP` | 150 | 相邻分块重叠长度 |

**调参建议**:
- 召回率偏低 → 增大 `DENSE_TOP_K` / `SPARSE_TOP_K`
- 首条结果不准 → 检查 Reranker 是否加载成功，或调大 `RERANK_TOP_K`
- 答案碎片化 → 增大 `CHUNK_SIZE`，减小 `CHUNK_OVERLAP`

### 学科名称映射

ChromaDB collection 名只能含 ASCII，因此做了映射：

| 学科 | Collection 名 |
|------|--------------|
| 数学 | `exam_math` |
| 英语 | `exam_english` |
| 政治 | `exam_politics` |
| 自动控制原理 | `exam_control` |

---

## RAG 评估

项目内置了完整的 RAG 评估框架，位于 `backend/eval/`。

### 评估指标

| 类别 | 指标 | 含义 | 是否需要标注 |
|------|------|------|-------------|
| **检索** | `Recall@K` | 相关文档被召回的比例 | ✅ |
| | `Precision@K` | 检索结果中相关文档占比 | ✅ |
| | `HitRate@K` | Top-K 中是否命中至少 1 个相关文档 | ✅ |
| | `MRR` | 首个相关文档排名的倒数均值 | ✅ |
| **生成** | `Faithfulness` | 答案 claims 被上下文支持的比例 | ❌ |
| | `Answer Relevance` | 答案是否直接回答问题 | ❌ |
| | `Context Precision` | 检索到的 chunks 中相关占比 | ❌ |

### 使用方式

```bash
# 步骤 1: 自动生成测试集（从向量库采样 chunks，LLM 出题）
python backend/eval/generate_testset.py --samples 20 --questions 1

# 步骤 2: 只评估检索指标（速度快，纯本地）
python backend/eval/evaluate.py --testset backend/eval/testset/testset.json --retrieval-only

# 步骤 3: 评估生成质量（需要 LLM 调用，建议采样）
python backend/eval/evaluate.py --testset backend/eval/testset/testset.json --generation-only --sample 10
```

### 指标解读

| 指标 | 优秀 | 及格 | 优化方向 |
|------|------|------|---------|
| Recall@5 | ≥ 0.8 | ≥ 0.5 | 增大 top_k，换更强 embedding |
| MRR | ≥ 0.5 | ≥ 0.3 | 调优 Reranker，检查 RRF 参数 |
| HitRate@1 | ≥ 0.3 | ≥ 0.15 | 优化 Reranker 排序 |
| Faithfulness | ≥ 0.8 | ≥ 0.6 | 加强 prompt 约束，增加上下文长度 |
| Answer Relevance | ≥ 0.7 | ≥ 0.5 | 改进检索精度 |

---

## 性能优化

### 1. Embedding 加速

当前 M3E 和 BGE-Reranker 在 CPU 上运行。如有 GPU：

```ini
# .env
DEVICE=cuda
```

确保安装 CUDA 版 PyTorch：
```bash
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. 模型本地化

将模型下载到本地路径，避免每次从 HuggingFace 下载：

```ini
M3E_MODEL_PATH=E:\models\m3e-base
BGE_RERANKER_PATH=E:\models\bge-reranker-base
```

### 3. MinerU 解析优化

- **本地版**: 首次下载模型后后续解析很快，无需网络
- **云端版**: 适合没有 GPU 的环境，但大文件（>200MB）会失败，自动降级到 PyMuPDF

### 4. 索引构建速度

| 场景 | 行为 | 时间 |
|------|------|------|
| 已有索引，无新文件 | 秒级返回"无需更新" | < 1s |
| 新增 1 个 PDF (10MB) | 只解析这 1 个文件 | 1~5 min (MinerU 云) |
| 新增 1 个 TXT/MD | 立即处理 | < 5s |
| 强制重建全部 | 清空并处理所有文件 | 取决于文件数量和大小 |

---

## 常见问题

**Q: 启动时报 `No module named 'torch'`？**
> 安装 PyTorch：`pip install torch transformers sentence-transformers`

**Q: 索引构建时提示"MinerU 云端解析失败: file size exceeds limit"？**
> 文件超过 200MB，自动降级到 PyMuPDF 解析。建议将大 PDF 拆分为小文件。

**Q: 点击"重建索引"后之前的问题还能搜到吗？**
> 可以。默认是**增量模式**，不清空已有数据，只添加新文件。

**Q: 如何强制清空重建？**
> 目前需手动删除 `vector_db/index_state.json` 后调用索引接口。前端"强制重建"开关后续可添加。

**Q: LLM 返回 401？**
> 检查 `.env` 中对应提供商的 API Key 是否正确。

**Q: BGE-Reranker 加载失败？**
> 设置 HuggingFace 镜像：`set HF_ENDPOINT=https://hf-mirror.com`，或手动下载模型到本地路径。

**Q: 公式显示乱码？**
> 前端已做 LaTeX 保护处理：`$...$` 和 `$$...$$` 内的内容不会经过 Markdown 解析，确保 KaTeX 正确渲染。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 向量数据库 | ChromaDB (持久化) |
| 稠密 Embedding | M3E-base (SentenceTransformers) |
| 稀疏检索 | BM25Okapi (rank-bm25) + jieba |
| 重排序 | BGE-Reranker (Cross-Encoder) |
| Agent 框架 | ReAct (自研，检索 + 联网工具) |
| LLM 协议 | OpenAI / Anthropic 兼容 |
| 前端 | 原生 HTML5 + CSS3 + JS (Marked.js + KaTeX + Highlight.js) |
| 传输协议 | SSE (Server-Sent Events) 流式输出 |

## License

MIT License
