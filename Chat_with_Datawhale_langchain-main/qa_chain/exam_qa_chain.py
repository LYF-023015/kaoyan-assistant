#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
考研智能题库问答链
支持题型识别、难度分级、知识点标注等功能
"""

import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from qa_chain.QA_chain_self import QA_chain_self
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.get_vectordb import get_vectordb
from qa_chain.model_to_llm import model_to_llm

class QuestionType:
    """题型枚举"""
    SINGLE_CHOICE = "单选题"
    MULTIPLE_CHOICE = "多选题"
    JUDGE = "判断题"
    FILL_BLANK = "填空题"
    SHORT_ANSWER = "简答题"
    ESSAY = "论述题"
    CALCULATION = "计算题"
    PROOF = "证明题"
    ANALYSIS = "分析题"

class DifficultyLevel:
    """难度等级"""
    EASY = "简单"
    MEDIUM = "中等"
    HARD = "困难"

class ExamQuestion:
    """考研题目类"""
    def __init__(self, content: str, question_type: str = None, difficulty: str = None,
                 subject: str = "全部", knowledge_points: List[str] = None):
        self.content = content
        self.question_type = question_type
        self.difficulty = difficulty
        self.subject = subject
        self.knowledge_points = knowledge_points or []
        self.created_at = datetime.now()

    def to_dict(self):
        return {
            'content': self.content,
            'question_type': self.question_type,
            'difficulty': self.difficulty,
            'subject': self.subject,
            'knowledge_points': self.knowledge_points,
            'created_at': self.created_at.isoformat()
        }

class ExamQAChain:
    """考研专用问答链，增强功能"""

    def __init__(self, model: str = "chatglm_std", embedding: str = "m3e",
                 temperature: float = 0.0, top_k: int = 4):
        self.model = model
        self.embedding = embedding
        self.temperature = temperature
        self.top_k = top_k
        self.qa_chain = None
        self.chat_qa_chain = None
        self.exam_db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exam_knowledge_db")
        self.persist_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exam_vector_db", "chroma")

        # 题型识别关键词
        self.type_keywords = {
            QuestionType.SINGLE_CHOICE: ["下列", "最", "正确的是", "错误的是", "属于", "不属于"],
            QuestionType.MULTIPLE_CHOICE: ["多项选择", "至少", "以下", "正确", "错误"],
            QuestionType.JUDGE: ["判断", "正确", "错误", "对", "错"],
            QuestionType.FILL_BLANK: ["填空", "横线", "空白", "填充"],
            QuestionType.SHORT_ANSWER: ["简述", "简答", "简要说明", "简析"],
            QuestionType.ESSAY: ["论述", "分析", "阐述", "谈谈", "你的看法"],
            QuestionType.CALCULATION: ["计算", "求", "求解", "化简", "证明"],
            QuestionType.PROOF: ["证明", "推导", "验证", "推导过程"],
            QuestionType.ANALYSIS: ["分析", "比较", "评价", "讨论", "说明"]
        }

        # 难度关键词权重
        self.difficulty_keywords = {
            DifficultyLevel.EASY: ["基础", "基本", "简单", "容易", "概念", "定义"],
            DifficultyLevel.MEDIUM: ["理解", "应用", "分析", "比较", "说明"],
            DifficultyLevel.HARD: ["综合", "复杂", "深入", "推导", "证明", "设计"]
        }

    def analyze_question_type(self, question: str) -> str:
        """分析题型"""
        question_lower = question.lower()

        # 特殊模式匹配
        if re.search(r'[a-d]\)|①|②|③|④', question):
            if re.search(r'多项|多选|至少两个', question):
                return QuestionType.MULTIPLE_CHOICE
            else:
                return QuestionType.SINGLE_CHOICE

        if re.search(r'判断|正确|错误|对|错', question):
            return QuestionType.JUDGE

        if re.search(r'填空|横线|空白', question):
            return QuestionType.FILL_BLANK

        if re.search(r'计算|求|化简|数值', question):
            return QuestionType.CALCULATION

        if re.search(r'证明|推导|验证', question):
            return QuestionType.PROOF

        # 基于关键词匹配
        type_scores = {}
        for qtype, keywords in self.type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            type_scores[qtype] = score

        # 返回得分最高的题型
        if type_scores:
            return max(type_scores, key=type_scores.get)

        return QuestionType.SHORT_ANSWER

    def analyze_difficulty(self, question: str) -> str:
        """分析难度"""
        question_lower = question.lower()

        # 计算各难度级别的得分
        difficulty_scores = {}
        for level, keywords in self.difficulty_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            difficulty_scores[level] = score

        # 基于题型调整难度
        question_type = self.analyze_question_type(question)
        if question_type in [QuestionType.CALCULATION, QuestionType.PROOF]:
            difficulty_scores[DifficultyLevel.HARD] += 2
        elif question_type in [QuestionType.SINGLE_CHOICE, QuestionType.JUDGE]:
            difficulty_scores[DifficultyLevel.EASY] += 1

        # 返回得分最高的难度
        if difficulty_scores:
            return max(difficulty_scores, key=difficulty_scores.get)

        return DifficultyLevel.MEDIUM

    def extract_knowledge_points(self, question: str, subject: str) -> List[str]:
        """提取知识点"""
        # 基于题型的知识点库
        knowledge_base = {
            "政治": ["马克思主义基本原理", "毛泽东思想", "中国特色社会主义", "思想道德修养", "中国近现代史"],
            "数学": ["高等数学", "线性代数", "概率论", "微积分", "方程", "函数"],
            "英语": ["词汇", "语法", "阅读理解", "写作", "翻译"],
            "自动控制原理": ["传递函数", "稳定性", "根轨迹", "频率响应", "PID控制", "状态空间"],
            "全部": ["基础知识", "理论概念", "应用方法", "计算技巧", "分析思路"]
        }

        # 根据学科和题型提取
        question_words = re.findall(r'[\u4e00-\u9fa5]+', question)

        possible_points = knowledge_base.get(subject, knowledge_base["全部"])
        matched_points = []

        for point in possible_points:
            if any(word in point for word in question_words):
                matched_points.append(point)

        # 如果没有匹配到，返回题型相关的通用知识点
        if not matched_points:
            question_type = self.analyze_question_type(question)
            type_knowledge = {
                QuestionType.CALCULATION: ["计算方法", "解题步骤", "公式应用"],
                QuestionType.PROOF: ["理论证明", "逻辑推理", "定理应用"],
                QuestionType.ANALYSIS: ["分析方法", "比较对比", "评价标准"],
                QuestionType.SINGLE_CHOICE: ["概念理解", "基本定义", "核心要点"],
                "default": ["基本概念", "核心理论", "应用方法"]
            }
            matched_points = type_knowledge.get(question_type, type_knowledge["default"])

        return matched_points[:3]  # 返回最多3个知识点

    def create_qa_chain(self, subject: str = "全部", use_history: bool = False):
        """创建问答链"""
        file_path = os.path.join(self.exam_db_path, subject) if subject != "全部" else self.exam_db_path
        persist_path = os.path.join(self.persist_path, subject) if subject != "全部" else self.persist_path

        try:
            # 获取向量数据库
            vectordb = get_vectordb(file_path, persist_path, self.embedding)

            if use_history:
                self.chat_qa_chain = Chat_QA_chain_self(
                    model=self.model,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    file_path=file_path,
                    persist_path=persist_path,
                    embedding=self.embedding,
                    subject=subject
                )
            else:
                self.qa_chain = QA_chain_self(
                    model=self.model,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    file_path=file_path,
                    persist_path=persist_path,
                    embedding=self.embedding,
                    subject=subject
                )

            return True
        except Exception as e:
            print(f"创建问答链失败: {str(e)}")
            return False

    def answer_question(self, question: str, subject: str = "全部",
                       use_history: bool = False, analyze: bool = True) -> Dict:
        """回答问题，可选择是否分析题目特征"""

        result = {
            'question': question,
            'answer': '',
            'analysis': {} if analyze else None,
            'similar_questions': []
        }

        # 分析题目特征
        if analyze:
            result['analysis'] = {
                'question_type': self.analyze_question_type(question),
                'difficulty': self.analyze_difficulty(question),
                'knowledge_points': self.extract_knowledge_points(question, subject),
                'subject': subject
            }

        # 创建问答链
        if not self.qa_chain and not self.chat_qa_chain:
            if not self.create_qa_chain(subject, use_history):
                result['error'] = "无法创建问答链，请确保知识库已正确初始化"
                return result

        # 调用问答
        try:
            if use_history:
                # 需要先同步历史记录
                if self.chat_qa_chain:
                    answer = self.chat_qa_chain.answer(question)
                    result['answer'] = answer[-1][1] if answer else "未能生成答案"
            else:
                if self.qa_chain:
                    answer = self.qa_chain.answer(question)
                    result['answer'] = answer

            # 找出相似题目
            result['similar_questions'] = self.find_similar_questions(question, subject)

        except Exception as e:
            result['error'] = f"回答问题时出错: {str(e)}"

        return result

    def find_similar_questions(self, question: str, subject: str = "全部") -> List[Dict]:
        """查找相似题目"""
        try:
            # 使用问题本身作为检索向量
            file_path = os.path.join(self.exam_db_path, subject) if subject != "全部" else self.exam_db_path
            persist_path = os.path.join(self.persist_path, subject) if subject != "全部" else self.persist_path

            vectordb = get_vectordb(file_path, persist_path, self.embedding)

            # 检索相似文档
            retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={'k': 3})
            docs = retriever.get_relevant_documents(question)

            similar_questions = []
            for doc in docs:
                # 从文档内容中提取可能的题目
                content = doc.page_content[:500]  # 只取前500字符
                if len(content) > 50:  # 过滤太短的内容
                    similar_questions.append({
                        'content': content,
                        'source': doc.metadata.get('source', 'unknown'),
                        'score': doc.metadata.get('score', 0)
                    })

            return similar_questions[:3]  # 返回最多3个相似题目

        except Exception as e:
            print(f"查找相似题目失败: {str(e)}")
            return []

    def generate_practice_set(self, subject: str, question_type: str = None,
                             difficulty: str = None, count: int = 10) -> List[Dict]:
        """生成练习题集"""
        practice_set = []

        # 这里应该从知识库中抽取符合条件的题目
        # 由于我们目前没有存储题目的数据库，这里返回示例题目

        sample_questions = {
            "政治": [
                ("简述马克思主义中国化时代化的重大理论成果。", QuestionType.SHORT_ANSWER, DifficultyLevel.MEDIUM),
                ("如何理解'两个确立'的决定性意义？", QuestionType.ESSAY, DifficultyLevel.HARD)
            ],
            "数学": [
                ("求极限：lim(x→0) (sinx - x) / x^3", QuestionType.CALCULATION, DifficultyLevel.MEDIUM),
                ("证明：如果级数∑an收敛，则lim(n→∞)an = 0。", QuestionType.PROOF, DifficultyLevel.HARD)
            ],
            "自动控制原理": [
                ("什么是系统的稳定性？劳斯判据如何判断系统稳定性？", QuestionType.SHORT_ANSWER, DifficultyLevel.MEDIUM),
                ("已知开环传递函数G(s) = K / [s(s+1)(s+2)]，求使系统稳定的K值范围。", QuestionType.CALCULATION, DifficultyLevel.HARD)
            ]
        }

        if subject in sample_questions:
            questions = sample_questions[subject]
            for content, qtype, diff in questions[:count]:
                question = ExamQuestion(content, qtype, diff, subject)
                practice_set.append(question.to_dict())

        return practice_set

    def get_study_statistics(self) -> Dict:
        """获取学习统计信息"""
        # 这里应该从数据库中获取实际统计信息
        # 目前返回示例数据

        return {
            'total_questions': 0,
            'by_subject': {
                '政治': 0,
                '英语': 0,
                '数学': 0,
                '控制工程': 0
            },
            'by_type': {
                QuestionType.SINGLE_CHOICE: 0,
                QuestionType.MULTIPLE_CHOICE: 0,
                QuestionType.SHORT_ANSWER: 0,
                QuestionType.CALCULATION: 0,
                QuestionType.PROOF: 0
            },
            'by_difficulty': {
                DifficultyLevel.EASY: 0,
                DifficultyLevel.MEDIUM: 0,
                DifficultyLevel.HARD: 0
            }
        }


# 测试代码
if __name__ == "__main__":
    chain = ExamQAChain()

    # 测试题型识别
    test_question = "下列选项中，属于马克思主义基本原理的是："
    print(f"题目: {test_question}")
    print(f"识别题型: {chain.analyze_question_type(test_question)}")
    print(f"识别难度: {chain.analyze_difficulty(test_question)}")
    print(f"知识点: {chain.extract_knowledge_points(test_question, '政治')}")