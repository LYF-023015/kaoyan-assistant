# 导入必要的库

import sys
import os                # 用于操作系统相关的操作，例如读取环境变量

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# import IPython.display   # 已注释 - 在普通 Python 环境中不需要，且代码中未使用
import io                # 用于处理流式数据（例如文件流）
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from llm.call_llm import get_completion
from database.create_db import create_db_info
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.QA_chain_self import QA_chain_self
import re
# 导入 dotenv 库的函数
# dotenv 允许您从 .env 文件中读取环境变量
# 这在开发时特别有用，可以避免将敏感信息（如API密钥）硬编码到代码中

# 寻找 .env 文件并加载它的内容
# 这允许您使用 os.environ 来读取在 .env 文件中设置的环境变量
_ = load_dotenv(find_dotenv())
LLM_MODEL_DICT = {
    "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
    "wenxin": ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"],
    "xinhuo": ["Spark-1.5", "Spark-2.0"],
    "zhipuai": ["chatglm_pro", "chatglm_std", "chatglm_lite"]
}


LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()),[])
INIT_LLM = "chatglm_std"
EMBEDDING_MODEL_LIST = ['zhipuai', 'openai', 'm3e']
INIT_EMBEDDING_MODEL = "m3e"

# 考研学科配置
SUBJECT_LIST = ["政治", "英语", "数学", "自动控制原理", "全部"]
INIT_SUBJECT = "全部"
SCHOOL_NAME = "南京理工大学"
MAJOR_NAME = "控制工程"

# 获取当前脚本所在目录的上一级目录（即项目根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DB_PATH = os.path.join(BASE_DIR, "exam_knowledge_db")
DEFAULT_PERSIST_PATH = os.path.join(BASE_DIR, "exam_vector_db", "chroma")
AIGC_AVATAR_PATH = os.path.join(BASE_DIR, "figures", "aigc_avatar.png")
DATAWHALE_AVATAR_PATH = os.path.join(BASE_DIR, "figures", "datawhale_avatar.png")
AIGC_LOGO_PATH = os.path.join(BASE_DIR, "figures", "aigc_logo.png")
DATAWHALE_LOGO_PATH = os.path.join(BASE_DIR, "figures", "datawhale_logo.png")

# 考研专用路径
EXAM_DB_PATH = os.path.join(BASE_DIR, "exam_knowledge_db")
EXAM_PERSIST_PATH = os.path.join(BASE_DIR, "exam_vector_db", "chroma")

def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "")

def get_subject_path(subject):
    """获取学科对应的知识库路径"""
    if subject == "全部":
        return EXAM_DB_PATH
    else:
        return os.path.join(EXAM_DB_PATH, subject)

class Model_center():
    """
    存储问答 Chain 的对象 

    - chat_qa_chain_self: 以 (model, embedding) 为键存储的带历史记录的问答链。
    - qa_chain_self: 以 (model, embedding) 为键存储的不带历史记录的问答链。
    """
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def chat_qa_chain_self_answer(self, question: str, chat_history: list = [], model: str = "openai", embedding: str = "openai", temperature: float = 0.0, top_k: int = 4, history_len: int = 3, file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH, subject: str = "全部"):
        """
        调用带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history_tuples = history_to_tuples(chat_history)
            if (model, embedding, subject) not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[(model, embedding, subject)] = Chat_QA_chain_self(model=model, temperature=temperature,
                                                                                    top_k=top_k, chat_history=chat_history_tuples, file_path=file_path, persist_path=persist_path, embedding=embedding, subject=subject)
            chain = self.chat_qa_chain_self[(model, embedding, subject)]
            chain = self.chat_qa_chain_self[(model, embedding, subject)]
            chain.chat_history = chat_history_tuples # sync history before asking
            ans_tuples = chain.answer(question=question, temperature=temperature, top_k=top_k)
            return "", tuples_to_history(ans_tuples)
        except Exception as e:
            return e, chat_history

    def qa_chain_self_answer(self, question: str, chat_history: list = [], model: str = "openai", embedding="openai", temperature: float = 0.0, top_k: int = 4, file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH, subject: str = "全部"):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding, subject) not in self.qa_chain_self:
                self.qa_chain_self[(model, embedding, subject)] = QA_chain_self(model=model, temperature=temperature,
                                                                       top_k=top_k, file_path=file_path, persist_path=persist_path, embedding=embedding, subject=subject)
            chain = self.qa_chain_self[(model, embedding, subject)]
            chain = self.qa_chain_self[(model, embedding, subject)]
            
            chat_history_tuples = history_to_tuples(chat_history)
            chat_history_tuples.append(
                (question, chain.answer(question, temperature, top_k)))
            return "", tuples_to_history(chat_history_tuples)
        except Exception as e:
            return e, chat_history

    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()


def get_user_content(msg):
    if isinstance(msg, dict):
        return msg.get("content", "")
    elif hasattr(msg, "content"):
        return msg.content
    return str(msg)

def history_to_tuples(chat_history):
    tuples = []
    for i in range(0, len(chat_history)-1, 2):
        user_msg = get_user_content(chat_history[i])
        bot_msg = get_user_content(chat_history[i+1])
        tuples.append((user_msg, bot_msg))
    return tuples

def tuples_to_history(tuples):
    history = []
    for user_msg, bot_msg in tuples:
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": bot_msg})
    return history

def format_chat_prompt(message, chat_history):
    """
    该函数用于格式化聊天 prompt。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    prompt: 格式化后的 prompt。
    """
    # 初始化一个空字符串，用于存放格式化后的聊天 prompt。
    prompt = ""
    # 遍历聊天历史记录。
    chat_history_tuples = history_to_tuples(chat_history)
    for turn in chat_history_tuples:
        # 从聊天记录中提取用户和机器人的消息。
        user_message, bot_message = turn
        # 更新 prompt，加入用户和机器人的消息。
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # 将当前的用户消息也加入到 prompt中，并预留一个位置给机器人的回复。
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    # 返回格式化后的 prompt。
    return prompt



def respond(message, chat_history, llm, history_len=3, temperature=0.1, max_tokens=2048):
    """
    该函数用于生成机器人的回复。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    "": 空字符串表示没有内容需要显示在界面上，可以替换为真正的机器人回复。
    chat_history: 更新后的聊天历史记录
    """
    if message == None or len(message) < 1:
            return "", chat_history
    try:
        # 限制 history 的记忆长度
        chat_history = chat_history[-history_len*2:] if history_len > 0 else []
        # 调用上面的函数，将用户的消息和聊天历史记录格式化为一个 prompt。
        formatted_prompt = format_chat_prompt(message, chat_history)
        # 使用llm对象的predict方法生成机器人的回复（注意：llm对象在此代码中并未定义）。
        bot_message = get_completion(
            formatted_prompt, llm, temperature=temperature, max_tokens=max_tokens)
        # 将bot_message中\n换为<br/>
        bot_message = re.sub(r"\\n", '<br/>', bot_message)
        # 将用户的消息和机器人的回复加入到聊天历史记录中。
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        # 返回一个空字符串和更新后的聊天历史记录（这里的空字符串可以替换为真正的机器人回复，如果需要显示在界面上）。
        return "", chat_history
    except Exception as e:
        return e, chat_history


model_center = Model_center()

CSS = """
body { font-family: 'Microsoft YaHei', sans-serif; }
.header-box {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    border-radius: 16px; padding: 24px; margin-bottom: 16px; text-align: center;
}
.header-box h1 { color: white; font-size: 26px; margin: 0 0 6px 0; }
.header-box p { color: #bfdbfe; font-size: 14px; margin: 0; }
.chat-panel { border-radius: 12px; border: 1px solid #e5e7eb; }
.side-panel { background: #f8fafc; border-radius: 12px; padding: 12px; border: 1px solid #e5e7eb; }
.btn-primary { background: #2563eb !important; border-radius: 8px !important; }
.btn-secondary { border-radius: 8px !important; }
.subject-tag { display: inline-block; padding: 4px 12px; border-radius: 20px;
    background: #dbeafe; color: #1d4ed8; font-size: 12px; margin: 2px; }
footer { display: none !important; }
"""

block = gr.Blocks(title="考研助手", css=CSS)
with block as demo:
    gr.HTML("""
    <div class="header-box">
        <h1>🎓 南京理工大学控制工程考研助手</h1>
        <p>基于 RAG 检索增强生成 · 智能问答 · 全程备考陪伴</p>
        <div style="margin-top:10px">
            <span class="subject-tag">📖 政治</span>
            <span class="subject-tag">🔤 英语</span>
            <span class="subject-tag">📐 数学</span>
            <span class="subject-tag">⚙️ 自动控制原理</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=4, elem_classes="chat-panel"):
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                placeholder="<center><h3>👋 你好！我是考研助手</h3><p>请在右侧选择学科，然后输入问题开始提问</p></center>"
            )
            msg = gr.Textbox(
                label="",
                placeholder="✏️ 输入考研问题，Enter 发送...",
                lines=2, max_lines=5
            )
            with gr.Row():
                db_with_his_btn = gr.Button("📚 知识库问答（带历史）", variant="primary", elem_classes="btn-primary")
                db_wo_his_btn = gr.Button("🔍 知识库问答（无历史）", variant="secondary", elem_classes="btn-secondary")
                llm_btn = gr.Button("💬 直接问LLM", variant="secondary", elem_classes="btn-secondary")
            with gr.Row():
                clear = gr.ClearButton(components=[chatbot], value="🗑️ 清空对话")

        with gr.Column(scale=1, elem_classes="side-panel"):
            gr.Markdown("#### 📂 知识库管理")
            file = gr.File(
                label='上传考研资料（支持PDF/TXT/MD/DOCX）',
                file_count='directory',
                file_types=['.txt', '.md', '.docx', '.pdf']
            )
            init_db = gr.Button("⚡ 初始化知识库", variant="primary")

            gr.Markdown("#### 🎯 学科")
            subject_select = gr.Dropdown(
                SUBJECT_LIST, label="", value=INIT_SUBJECT, interactive=True
            )

            with gr.Accordion("⚙️ 参数", open=False):
                temperature = gr.Slider(0, 1, value=0.01, step=0.01, label="温度")
                top_k = gr.Slider(1, 10, value=3, step=1, label="Top-K")
                history_len = gr.Slider(0, 5, value=3, step=1, label="历史轮数")

            with gr.Accordion("🤖 模型", open=False):
                llm = gr.Dropdown(LLM_MODEL_LIST, label="大语言模型", value=INIT_LLM, interactive=True)
                embeddings = gr.Dropdown(EMBEDDING_MODEL_LIST, label="嵌入模型", value=INIT_EMBEDDING_MODEL)

        # 设置初始化向量数据库按钮的点击事件
        init_db.click(create_db_info,
                      inputs=[file, embeddings, subject_select], outputs=[msg])

        # 设置按钮的点击事件，传入学科选择
        db_with_his_btn.click(model_center.chat_qa_chain_self_answer, inputs=[
                              msg, chatbot, llm, embeddings, temperature, top_k, history_len,
                              gr.State(DEFAULT_DB_PATH), gr.State(DEFAULT_PERSIST_PATH), subject_select],
                              outputs=[msg, chatbot])

        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot, llm, embeddings, temperature, top_k,
                            gr.State(DEFAULT_DB_PATH), gr.State(DEFAULT_PERSIST_PATH), subject_select],
                            outputs=[msg, chatbot])

        llm_btn.click(respond, inputs=[
                      msg, chatbot, llm, history_len, temperature], outputs=[msg, chatbot], show_progress="minimal")

        # 设置文本框的提交事件
        msg.submit(respond, inputs=[
                   msg, chatbot, llm, history_len, temperature], outputs=[msg, chatbot], show_progress="hidden")

        # 点击后清空后端存储的聊天记录
        clear.click(model_center.clear_history)
    gr.HTML("""
    <div style="text-align:center; padding:12px; background:#f1f5f9; border-radius:8px; margin-top:8px; color:#64748b; font-size:13px">
        ① 上传资料 → 初始化知识库 &nbsp;|&nbsp; ② 选择学科 &nbsp;|&nbsp; ③ 输入问题 → 知识库问答
    </div>
    """)
# threads to consume the request
gr.close_all()
demo.launch(theme=gr.themes.Soft())
