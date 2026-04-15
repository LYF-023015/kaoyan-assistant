#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
增强版考研资料网络爬虫
支持从多个数据源下载考研资料，并自动进行向量化处理
"""

import os
import re
import json
import time
import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote
import hashlib
from datetime import datetime
import warnings
from typing import List, Dict, Optional, Set
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class EnhancedExamCrawler:
    """增强版考研资料爬虫类"""

    def __init__(self, base_path=None, embedding_model="m3e"):
        """
        初始化爬虫

        Args:
            base_path: 基础路径，默认为脚本所在目录的上层
            embedding_model: 使用的嵌入模型
        """
        if base_path is None:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.base_path = base_path
        self.exam_db_path = os.path.join(base_path, "exam_knowledge_db")
        self.download_dir = os.path.join(self.exam_db_path, "downloads")
        self.embedding_model = embedding_model

        # 学科映射
        self.subject_map = {
            "政治": "政治",
            "英语": "英语",
            "数学": "数学",
            "控制工程": "控制工程"
        }

        # 爬取配置
        self.max_retries = 3
        self.request_delay = 1.0
        self.timeout = 30
        self.concurrent_limit = 5

        # 确保目录存在
        self._ensure_directories()

        # 导入嵌入模块
        try:
            from embedding.call_embedding import get_embedding
            self.embedding_func = get_embedding(embedding_model)
            logger.info(f"使用嵌入模型: {embedding_model}")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            self.embedding_func = None

    def _ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(self.download_dir, exist_ok=True)
        for subject in self.subject_map.values():
            os.makedirs(os.path.join(self.exam_db_path, subject), exist_ok=True)

    def get_safe_filename(self, filename: str) -> str:
        """生成安全的文件名"""
        # 移除特殊字符
        filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        filename = filename.replace(" ", "_")
        # 限制长度
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        return filename

    async def download_content(self, session: aiohttp.ClientSession, url: str,
                            filename: str, is_pdf: bool = False) -> Optional[str]:
        """异步下载内容"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        for attempt in range(self.max_retries):
            try:
                async with session.get(url, headers=headers, timeout=self.timeout) as response:
                    response.raise_for_status()

                    content = await response.read()
                    filepath = os.path.join(self.download_dir, filename)

                    if is_pdf:
                        # 保存PDF
                        async with aiofiles.open(filepath, 'wb') as f:
                            await f.write(content)
                    else:
                        # 保存文本，尝试检测编码
                        try:
                            text = content.decode('utf-8')
                        except UnicodeDecodeError:
                            text = content.decode('gbk', errors='ignore')

                        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                            await f.write(text)

                    logger.info(f"成功下载: {filename}")
                    return filepath

            except Exception as e:
                logger.warning(f"下载失败 (尝试 {attempt + 1}/{self.max_retries}): {url}, 错误: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.request_delay * 2)
                else:
                    logger.error(f"下载最终失败: {url}")
                    return None

        return None

    async def crawl_moe_syllabus(self, session: aiohttp.ClientSession, year: int = 2025) -> None:
        """爬取研招网考研资讯"""
        logger.info(f"开始爬取{year}年考研资讯...")

        base_url = "https://yz.chsi.com.cn"
        news_url = f"{base_url}/kyzx/kydt/"

        try:
            async with session.get(news_url) as response:
                response.raise_for_status()
                html = await response.text(encoding='utf-8', errors='ignore')
                soup = BeautifulSoup(html, 'html.parser')

                links = soup.find_all('a', href=True)
                count = 0

                for link in links:
                    href = link.get('href', '')
                    text = link.get_text().strip()
                    if len(text) > 5 and count < 5:
                        full_url = urljoin(base_url, href)
                        try:
                            async with session.get(full_url) as article_resp:
                                article_resp.raise_for_status()
                                article_html = await article_resp.text(encoding='utf-8', errors='ignore')
                                article_soup = BeautifulSoup(article_html, 'html.parser')
                                content = article_soup.get_text()
                                content = re.sub(r'\s+', ' ', content).strip()

                                safe_text = re.sub(r'[\\/*?:"<>|]', "", text)[:80]
                                filename = f"研招网_{safe_text}.txt"
                                filepath = os.path.join(self.exam_db_path, "政治", filename)
                                with open(filepath, 'w', encoding='utf-8') as f:
                                    f.write(f"来源：中国研究生招生信息网\n标题：{text}\n链接：{full_url}\n\n{content}")
                                logger.info(f"已保存: {filename}")
                                count += 1
                                await asyncio.sleep(self.request_delay)
                        except Exception as e:
                            logger.warning(f"下载文章失败: {e}")

        except Exception as e:
            logger.error(f"爬取研招网资讯失败: {e}")

    async def crawl_exam_papers(self, session: aiohttp.ClientSession) -> None:
        """从考研帮爬取历年真题"""
        import requests as req
        import concurrent.futures
        logger.info("开始从考研帮爬取历年真题...")

        base_url = "https://download.kaoyan.com"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': base_url,
            'Accept-Language': 'zh-CN,zh;q=0.9',
        }
        subject_pages = {
            "政治": f"{base_url}/list-subject-1.html",
            "英语": f"{base_url}/list-subject-2.html",
            "数学": f"{base_url}/list-subject-3.html",
        }

        def download_subject(subject, list_url):
            try:
                r = req.get(list_url, headers=headers, timeout=15)
                r.encoding = 'utf-8'
                soup = BeautifulSoup(r.text, 'html.parser')

                detail_links = []
                for a in soup.find_all('a', href=True):
                    href = a.get('href', '')
                    text = a.get_text().strip()
                    if '/download-' in href and len(text) > 5:
                        full = urljoin(base_url, href)
                        if full not in detail_links:
                            detail_links.append(full)
                    if len(detail_links) >= 3:
                        break

                for detail_url in detail_links:
                    try:
                        dr = req.get(detail_url, headers=headers, timeout=15)
                        dr.encoding = 'utf-8'
                        dsoup = BeautifulSoup(dr.text, 'html.parser')
                        for a in dsoup.find_all('a', href=True):
                            href = a.get('href', '')
                            if 'xiazai' in href:
                                xiazai_url = urljoin(base_url, href)
                                title = a.get_text().strip() or os.path.basename(href)
                                xr = req.get(xiazai_url, headers=headers, timeout=15)
                                xr.encoding = 'utf-8'
                                xsoup = BeautifulSoup(xr.text, 'html.parser')
                                form = xsoup.find('form', id='downLoad')
                                if form:
                                    dl_url = form.get('action', '')
                                    if dl_url:
                                        safe = re.sub(r'[\\/*?:"<>|]', '', title)[:80]
                                        filename = f"{subject}_{safe}.pdf"
                                        dest = os.path.join(self.exam_db_path, subject, filename)
                                        pdf_r = req.get(dl_url, headers=headers, timeout=30, stream=True)
                                        if pdf_r.status_code == 200 and len(pdf_r.content) > 1000:
                                            with open(dest, 'wb') as f:
                                                f.write(pdf_r.content)
                                            logger.info(f"已下载: {filename}")
                                        else:
                                            logger.warning(f"下载内容为空: {filename}")
                                break
                        time.sleep(self.request_delay)
                    except Exception as e:
                        logger.warning(f"下载{subject}真题失败: {e}")
            except Exception as e:
                logger.warning(f"爬取考研帮{subject}页面失败: {e}")

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            tasks = [
                loop.run_in_executor(pool, download_subject, subject, url)
                for subject, url in subject_pages.items()
            ]
            await asyncio.gather(*tasks)

    async def crawl_university_info(self, session: aiohttp.ClientSession) -> None:
        """爬取目标院校信息"""
        logger.info("开始爬取目标院校信息...")

        # 目标院校列表
        universities = [
            {
                "name": "南京理工大学",
                "url": "https://gs.njust.edu.cn",
                "paths": [
                    "/zsgz/ksjj/kzgy.htm",
                    "/zsgz/ksjj/kybk.htm",
                    "/zsgz/ksjj/cjwt.htm",
                    "/jxpy/kjgc/kzgc.htm"
                ]
            }
        ]

        for university in universities:
            logger.info(f"正在爬取: {university['name']}")

            for path in university["paths"]:
                try:
                    url = urljoin(university["url"], path)
                    async with session.get(url) as response:
                        response.raise_for_status()
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')

                        # 提取主要内容
                        content = soup.get_text()

                        # 清理内容
                        content = re.sub(r'\s+', ' ', content).strip()
                        content = re.sub(r'[\n\r]+', '\n', content)

                        # 确定学科和文件名
                        if "政治" in path or "思政" in path:
                            subject = "政治"
                        elif "英语" in path or "en" in path.lower():
                            subject = "英语"
                        elif "数学" in path or "math" in path.lower():
                            subject = "数学"
                        else:
                            subject = "控制工程"

                        filename = f"{university['name']}_{os.path.basename(path)}.txt"
                        filepath = os.path.join(self.download_dir, filename)

                        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                            await f.write(content)

                        # 移动到对应学科目录
                        src_path = os.path.join(self.download_dir, filename)
                        if os.path.exists(src_path):
                            dest_path = os.path.join(self.exam_db_path, subject, filename)
                            os.rename(src_path, dest_path)

                        logger.info(f"成功下载: {filename}")
                        await asyncio.sleep(self.request_delay)

                except Exception as e:
                    logger.warning(f"爬取{university['name']}资料失败: {e}")

    def crawl_reference_books(self) -> None:
        """爬取参考教材信息"""
        logger.info("开始爬取参考教材信息...")

        # 教育部推荐教材列表
        books_info = {
            "政治": {
                "description": "政治学科推荐教材",
                "books": [
                    "马克思主义基本原理概论（高等教育出版社）",
                    "毛泽东思想和中国特色社会主义理论体系概论（高等教育出版社）",
                    "中国近现代史纲要（高等教育出版社）",
                    "思想道德与法治（高等教育出版社）",
                    "形势与政策（高等教育出版社）"
                ]
            },
            "英语": {
                "description": "英语学科推荐教材",
                "books": [
                    "考研英语词汇闪过（考研英语词汇书）",
                    "考研英语真相（考研英语真题解析）",
                    "考研英语写作高分攻略（考研英语写作书）",
                    "张剑黄皮书考研英语历年真题解析"
                ]
            },
            "数学": {
                "description": "数学学科推荐教材",
                "books": [
                    "高等数学（同济大学第七版）",
                    "线性代数（同济大学第六版）",
                    "概率论与数理统计（浙江大学第四版）",
                    "考研数学复习全书（李永乐、王式安）",
                    "考研数学历年真题解析"
                ]
            },
            "控制工程": {
                "description": "控制工程学科推荐教材",
                "books": [
                    "自动控制原理（胡寿松第七版）",
                    "现代控制理论（谢克明第三版）",
                    "计算机控制系统（李正军第二版）",
                    "过程控制工程（金以慧第三版）",
                    "考研自动控制原理真题解析"
                ]
            }
        }

        for subject, info in books_info.items():
            content = f"{info['description']}\n\n"
            content += "推荐教材列表：\n\n"

            for i, book in enumerate(info['books'], 1):
                content += f"{i}. {book}\n"

            content += "\n\n更新时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            filename = f"{subject}_推荐教材.txt"
            filepath = os.path.join(self.exam_db_path, subject, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"已生成{subject}教材列表")

    async def crawl_study_guides(self, session: aiohttp.ClientSession) -> None:
        """爬取考研学习指南"""
        logger.info("开始爬取考研学习指南...")

        # 学习资源网站
        study_sites = [
            {
                "name": "中国研招网",
                "url": "https://yz.chsi.com.cn",
                "path": "/kyzx/kydt/"
            }
        ]

        for site in study_sites:
            try:
                url = urljoin(site["url"], site["path"])
                async with session.get(url) as response:
                    response.raise_for_status()
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # 查找文章列表
                    articles = soup.find_all('a', href=True)

                    for article in articles[:10]:  # 限制为10篇文章
                        text = article.get_text().strip()
                        href = article.get('href', '')

                        if len(text) > 10 and ('经验' in text or '攻略' in text or '指南' in text):
                            try:
                                full_url = urljoin(site["url"], href)
                                async with session.get(full_url) as article_response:
                                    article_response.raise_for_status()
                                    article_html = await article_response.text()
                                    article_soup = BeautifulSoup(article_html, 'html.parser')

                                    # 提取文章内容
                                    article_content = article_soup.get_text()

                                    # 确定学科
                                    subject = "全部"  # 默认为全部
                                    for sub in ["政治", "英语", "数学", "控制"]:
                                        if sub in text:
                                            subject = sub
                                            break

                                    # 保存文章
                                    safe_text = re.sub(r'[\\/*?:"<>|]', "", text)[:100]
                                    filename = f"{site['name']}_{safe_text}.txt"
                                    filepath = os.path.join(self.download_dir, filename)

                                    async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                                        await f.write(f"来源：{site['name']}\n")
                                        await f.write(f"标题：{text}\n")
                                        await f.write(f"链接：{full_url}\n")
                                        await f.write("-" * 50 + "\n\n")
                                        await f.write(article_content)

                                    # 移动到对应学科目录
                                    src_path = os.path.join(self.download_dir, filename)
                                    if os.path.exists(src_path):
                                        dest_path = os.path.join(self.exam_db_path, subject, filename)
                                        os.rename(src_path, dest_path)

                                    logger.info(f"已保存学习指南: {text}")
                                    await asyncio.sleep(self.request_delay)

                            except Exception as e:
                                logger.warning(f"下载文章失败: {text}, 错误: {e}")

            except Exception as e:
                logger.warning(f"爬取{site['name']}失败: {e}")

    def create_sample_questions(self) -> None:
        """创建示例题目"""
        logger.info("创建示例题目...")

        subjects = {
            "政治": [
                {
                    "question": "简述马克思主义中国化时代化的重大理论成果。",
                    "answer": "马克思主义中国化时代化的重大理论成果包括毛泽东思想、邓小平理论、'三个代表'重要思想、科学发展观和新时代中国特色社会主义思想。",
                    "type": "简答题",
                    "difficulty": "中等"
                },
                {
                    "question": "如何理解'两个确立'的决定性意义？",
                    "answer": "'两个确立'是指确立习近平同志党中央的核心、全党的核心地位，确立习近平新时代中国特色社会主义思想的指导地位。这是时代的选择、历史的选择、人民的选择。",
                    "type": "论述题",
                    "difficulty": "较难"
                }
            ],
            "数学": [
                {
                    "question": "求极限：lim(x→0) (sinx - x) / x^3",
                    "answer": "使用洛必达法则：\n原式 = lim(x→0) (cosx - 1) / (3x^2)\n= lim(x→0) (-sinx) / (6x)\n= lim(x→0) (-cosx) / 6\n= -1/6",
                    "type": "计算题",
                    "difficulty": "中等"
                },
                {
                    "question": "证明：如果级数∑an收敛，则lim(n→∞)an = 0。",
                    "answer": "证明：\n因为级数∑an收敛，根据级数收敛的必要条件，\n有lim(n→∞)Sn = S（收敛）\n其中Sn是前n项和。\n又因为Sn-1 = Sn - an\n取极限得：S = S - lim(n→∞)an\n所以lim(n→∞)an = 0",
                    "type": "证明题",
                    "difficulty": "较难"
                }
            ],
            "控制工程": [
                {
                    "question": "什么是系统的稳定性？劳斯判据如何判断系统稳定性？",
                    "answer": "系统稳定性：当系统受到扰动后，能够自动恢复到平衡状态的能力。\n\n劳斯判据步骤：\n1. 列出特征方程的各项系数\n2. 构造劳斯表\n3. 检查第一列元素\n4. 如果第一列元素全为正，系统稳定；否则系统不稳定，且符号变化次数为不稳定根的个数。",
                    "type": "概念题",
                    "difficulty": "中等"
                },
                {
                    "question": "已知开环传递函数G(s) = K / [s(s+1)(s+2)]，求使系统稳定的K值范围。",
                    "answer": "特征方程：1 + G(s) = 0\n即：s^3 + 3s^2 + 2s + K = 0\n\n劳斯表：\ns^3: 1  2\ns^2: 3  K\ns^1: (6-K)/3\ns^0: K\n\n稳定条件：\n(6-K)/3 > 0 且 K > 0\n即：0 < K < 6\n\n所以K的取值范围是(0,6)。",
                    "type": "计算题",
                    "difficulty": "较难"
                }
            ]
        }

        for subject, questions in subjects.items():
            content = f"{subject}示例题目\n\n"
            content += f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

            for i, q in enumerate(questions, 1):
                content += f"题目{i}：{q['question']}\n\n"
                content += f"参考答案：\n{q['answer']}\n\n"
                content += f"题型：{q['type']} | 难度：{q['difficulty']}\n"
                content += "-" * 50 + "\n\n"

            filename = f"{subject}_示例题目.txt"
            filepath = os.path.join(self.exam_db_path, subject, filename)

            # 使用同步方式写入文件
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"已生成{subject}示例题目")

    def import_local_files(self, local_path: str) -> None:
        """从本地文件夹导入资料到知识库"""
        if not os.path.exists(local_path):
            logger.warning(f"本地路径不存在: {local_path}")
            return
        logger.info(f"开始从本地导入文件: {local_path}")
        supported = ('.txt', '.md', '.pdf', '.docx')
        count = 0
        for root, _, files in os.walk(local_path):
            for fname in files:
                if not fname.lower().endswith(supported):
                    continue
                src = os.path.join(root, fname)
                # 根据文件名判断学科
                subject = "控制工程"
                for subj in ["政治", "英语", "数学"]:
                    if subj in fname or subj in root:
                        subject = subj
                        break
                dest_dir = os.path.join(self.exam_db_path, subject)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, fname)
                if not os.path.exists(dest):
                    import shutil
                    shutil.copy2(src, dest)
                    logger.info(f"已导入: {fname} -> {subject}")
                    count += 1
        logger.info(f"本地文件导入完成，共导入 {count} 个文件")

    async def crawl_all(self, local_path: str = None) -> None:
        """执行所有爬取任务"""
        logger.info("开始批量爬取考研资料...")

        # 创建异步会话
        connector = aiohttp.TCPConnector(limit=self.concurrent_limit)
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # 执行各个爬取任务
            tasks = []

            # 爬取考试大纲
            tasks.append(asyncio.create_task(self.crawl_moe_syllabus(session=session)))

            # 爬取历年真题
            tasks.append(asyncio.create_task(self.crawl_exam_papers(session=session)))

            # 爬取院校信息
            tasks.append(asyncio.create_task(self.crawl_university_info(session=session)))

            # 爬取学习指南
            tasks.append(asyncio.create_task(self.crawl_study_guides(session=session)))

            # 等待所有任务完成
            await asyncio.gather(*tasks)

        # 生成教材列表和示例题目（同步任务）
        self.crawl_reference_books()
        self.create_sample_questions()

        # 导入本地文件
        if local_path:
            self.import_local_files(local_path)

        logger.info("\n考研资料爬取完成！")
        logger.info(f"资料保存在: {self.exam_db_path}")

        # 显示文件统计
        for subject in self.subject_map.values():
            subject_path = os.path.join(self.exam_db_path, subject)
            if os.path.exists(subject_path):
                files = os.listdir(subject_path)
                logger.info(f"{subject}: {len(files)} 个文件")

    def crawl_sync(self, local_path: str = None) -> None:
        """同步方式运行爬虫"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.crawl_all(local_path=local_path))
        finally:
            loop.close()


if __name__ == "__main__":
    # 创建爬虫实例
    crawler = EnhancedExamCrawler()

    # 执行爬取
    crawler.crawl_sync()