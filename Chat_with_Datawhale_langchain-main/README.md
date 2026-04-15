# 南京理工大学控制工程考研助手

这是一个基于LangChain和Gradio构建的智能考研助手系统，可以自动从网上下载考研资料、创建向量数据库，并提供智能问答功能。

## 功能特点

1. **自动资料下载** - 从教育部网站、各大院校官网等自动下载考研大纲、真题、招生简章等资料
2. **智能文档处理** - 支持PDF、Word、Markdown等多种格式的文档处理和分割
3. **向量数据库** - 使用ChromaDB创建向量数据库，支持语义搜索
4. **多模型支持** - 支持多种嵌入模型（m3e、OpenAI、智谱AI等）
5. **定期更新** - 支持定时自动更新考研资料
6. **Web界面** - 提供友好的Gradio Web界面

## 系统要求

- Python 3.8+
- 4GB以上内存
- 网络连接（用于下载资料）

## 安装步骤

### 1. 克隆项目

```bash
git clone <项目地址>
cd Chat_with_Datawhale_langchain-main
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量（可选）

如果使用OpenAI或智谱AI的嵌入模型，请创建`.env`文件：

```
OPENAI_API_KEY=your_openai_api_key
ZHIPUAI_API_KEY=your_zhipuai_api_key
```

## 使用方法

### 首次使用

1. 运行启动脚本：
```bash
python start_exam_assistant.py
```

2. 首次启动选择"1. 首次启动（初始化+优化+运行）"

3. 系统会自动：
   - 爬取考研资料
   - 处理文档
   - 创建向量数据库
   - 启动Web界面

### 运行选项

```bash
# 仅运行Web界面
python start_exam_assistant.py --web

# 初始化数据库
python start_exam_assistant.py --init

# 运行数据更新器
python start_exam_assistant.py --update

# 指定嵌入模型
python start_exam_assistant.py --init --embedding openai
```

## Web界面功能

1. **知识库问答（带历史）** - 基于知识库回答问题，保留对话历史
2. **知识库问答（无历史）** - 基于知识库回答问题，不保留历史
3. **直接问LLM** - 直接调用大模型，不使用知识库
4. **学科选择** - 可选择特定学科或全部学科
5. **参数配置** - 可调整温度、top_k等参数
6. **上传资料** - 可手动上传考研资料文件

## 文件结构

```
Chat_with_Datawhale_langchain-main/
├── serve/                 # Web服务
│   ├── run_gradio.py      # Gradio主程序
│   └── api.py             # API接口
├── database/              # 数据库相关
│   ├── web_crawler.py     # 增强版爬虫
│   ├── create_db.py       # 数据库创建
│   └── call_embedding.py  # 嵌入模型调用
├── llm/                   # 大模型相关
│   ├── call_llm.py        # LLM调用
│   ├── openai_llm.py     # OpenAI实现
│   ├── zhipuai_llm.py    # 智谱AI实现
│   └── wenxin_llm.py     # 文心一言实现
├── qa_chain/              # 问答链
│   ├── Chat_QA_chain_self.py    # 带历史问答
│   └── QA_chain_self.py         # 无历史问答
├── utils/                 # 工具模块
│   ├── document_processor.py    # 文档处理器
│   ├── data_updater.py         # 数据更新器
│   └── study_plan.py           # 学习计划
├── exam_knowledge_db/     # 考研知识库
├── exam_vector_db/       # 向量数据库
├── figures/               # 图片资源
├── backups/              # 备份目录
├── start_exam_assistant.py    # 启动脚本
└── requirements.txt       # 依赖列表
```

## 自定义配置

### 1. 修改爬虫配置

编辑 `utils/data_updater.py` 中的配置：

```python
config = {
    "embedding_model": "m3e",  # 嵌入模型
    "crawl_interval": "weekly",  # 更新间隔
    "subjects": ["政治", "英语", "数学", "控制工程"],  # 学科列表
    "max_retries": 3,  # 最大重试次数
    "backup_enabled": True,  # 是否启用备份
}
```

### 2. 添加新的数据源

在 `database/web_crawler.py` 中添加新的爬取方法：

```python
async def crawl_new_source(self, session: aiohttp.ClientSession):
    """爬取新的数据源"""
    # 实现新的爬取逻辑
    pass
```

### 3. 修改文档处理参数

在 `utils/document_processor.py` 中调整分块参数：

```python
def __init__(self, embedding_model="m3e", chunk_size=1000, chunk_overlap=200):
    # 修改分块大小和重叠
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap
```

## 常见问题

### 1. Q: 爬取资料失败怎么办？
A: 检查网络连接，或手动下载资料后放到对应学科目录下。

### 2. Q: 如何更换嵌入模型？
A: 修改启动命令中的 `--embedding` 参数，或修改配置文件。

### 3. Q: 知识库如何更新？
A: 运行 `python start_exam_assistant.py --update` 手动更新，或设置定时更新。

### 4. Q: 如何添加新的学科？
A: 在 `subjects` 列表中添加新学科，并创建对应的目录。

### 5. Q: 服务器部署如何配置？
A: 设置环境变量 `PORT1` 指定端口，或修改 `run_gradio.py` 中的 launch 参数。

## 开发说明

### 添加新的数据源

1. 在 `web_crawler.py` 中添加异步爬取方法
2. 确保正确处理异常和重试
3. 添加适当的日志记录

### 自定义嵌入模型

1. 在 `embedding/call_embedding.py` 中添加新的嵌入函数
2. 在配置中添加模型选项
3. 更新文档处理器的初始化

### 扩展问答功能

1. 在 `qa_chain/` 目录下添加新的问答链
2. 在 `run_gradio.py` 中添加新的界面组件
3. 更新Model_center类以支持新的问答模式

## 许可证

本项目基于 MIT 许可证开源。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题请提交 Issue 或联系开发者。