from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
import sys
sys.path.append('/Users/lta/Desktop/llm-universe/project')
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
import re

class Chat_QA_chain_self:
    """"
    带历史记录的问答链  
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - chat_history：历史记录，输入一个列表，默认是一个空列表
    - history_len：控制保留的最近 history_len 次对话
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火
    - api_key：星火、百度文心、OpenAI、智谱都需要传递的参数
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）  
    """
    def __init__(self,model:str, temperature:float=0.0, top_k:int=4, chat_history:list=[], file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None,Wenxin_secret_key:str=None, embedding = "openai",embedding_key:str=None, subject:str="全部"):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        #self.history_len = history_len
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key
        self.subject = subject

        # 考研专用提示词模板
        self.exam_templates = {
            "全部": """你是一位专业的考研辅导老师，精通所有考研科目。请根据以下考研相关资料回答问题。
要求：
1. 回答要准确、简洁，重点突出
2. 对于概念性问题，要给出定义和例子
3. 对于计算题，要给出详细步骤
4. 最后一定要总结关键点

{context}
考研问题: {question}
专业的回答:""",
            "政治": """你是一位考研政治辅导专家。请根据以下政治学习资料回答问题。
回答要求：
1. 准确表述理论要点，引用马原、毛中特、史纲、思修相关内容
2. 结合时政热点进行分析
3. 答题要分点清晰，逻辑严谨
4. 最后用一句话总结核心观点

{context}
考研政治问题: {question}
政治学回答:""",
            "英语": """你是一位考研英语辅导名师。请根据以下英语学习资料回答问题。
回答要求：
1. 对于词汇问题，给出词义、用法搭配和例句
2. 对于阅读理解，分析文章结构和答题思路
3. 对于写作，提供高分句型和范文参考
4. 注意区分不同题型的答题技巧

{context}
考研英语问题: {question}
英语学习解答:""",
            "数学": """你是一位考研数学辅导专家。请根据以下数学学习资料回答问题。
回答要求：
1. 对于计算题，给出详细的解题步骤
2. 对于证明题，写出严谨的推理过程
3. 标注重要公式和定理
4. 提供解题技巧和易错点提醒

{context}
考研数学问题: {question}
数学解答:""",
            "自动控制原理": """你是一位南京理工大学控制工程专业的考研辅导老师。请根据以下控制原理资料回答问题。
回答要求：
1. 准确使用专业术语，公式表达规范
2. 画出必要的控制系统结构图或信号流图
3. 分析时结合南理工的出题特点
4. 提供典型例题和解题思路

{context}
南京理工大学控制工程问题: {question}
控制原理解答:"""
        }

        # 获取当前学科的提示词模板
        if subject in self.exam_templates:
            self.prompt_template = self.exam_templates[subject]
        else:
            self.prompt_template = self.exam_templates["全部"]

        # 创建自定义提示词
        self.custom_prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )


        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)
        
    
    def clear_history(self):
        "清空历史记录"
        return self.chat_history.clear()

    
    def change_history_length(self,history_len:int=1):
        """
        保存指定对话轮次的历史记录
        输入参数：
        - history_len ：控制保留的最近 history_len 次对话
        - chat_history：当前的历史对话记录
        输出：返回最近 history_len 次对话
        """
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]

 
    def answer(self, question:str=None,temperature = None, top_k = 4):
        """"
        核心方法，调用问答链
        arguments: 
        - question：用户提问
        """
        
        if len(question) == 0:
            return "", self.chat_history
        
        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature
        llm = model_to_llm(self.model, temperature, self.appid, self.api_key, self.Spark_api_secret,self.Wenxin_secret_key)

        retriever = self.vectordb.as_retriever(search_type="similarity",
                                        search_kwargs={'k': top_k})

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            combine_docs_chain_kwargs={"prompt": self.custom_prompt}
        )
        
        #print(self.llm)
        result = qa({"question": question,"chat_history": self.chat_history})       #result里有question、chat_history、answer
        answer =  result['answer']
        answer = re.sub(r"\\n", '<br/>', answer)
        self.chat_history.append((question,answer)) #更新历史记录

        return self.chat_history  #返回本次回答和更新后的历史记录
        
















