#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
考研助手启动脚本
一键启动考研知识助手系统
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def check_dependencies():
    """检查依赖是否安装"""
    print("检查依赖...")
    missing_deps = []
    required_packages = [
        ('gradio', 'gradio'),
        ('langchain', 'langchain'),
        ('langchain_community', 'langchain-community'),
        ('chromadb', 'chromadb'),
        ('bs4', 'beautifulsoup4'),
        ('pymupdf', 'PyMuPDF'),
        ('aiohttp', 'aiohttp'),
        ('schedule', 'schedule'),
        ('jieba', 'jieba'),
        ('openai', 'openai'),
        ('zhipuai', 'zhipuai')
    ]

    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_deps.append(package_name)

    if missing_deps:
        print(f"✗ 缺少依赖: {', '.join(missing_deps)}")
        print("请运行: pip install -r requirements.txt")
        return False

    print("✓ 所有依赖已安装")
    return True

def initialize_database(embedding_model="m3e", subjects=None, local_path=None):
    """初始化数据库"""
    if subjects is None:
        subjects = ["政治", "英语", "数学", "控制工程"]

    print(f"\n初始化考研知识库（嵌入模型: {embedding_model}）...")
    try:
        # 爬取资料
        print("1. 爬取考研资料...")
        from database.web_crawler import EnhancedExamCrawler
        crawler = EnhancedExamCrawler(embedding_model=embedding_model)
        crawler.crawl_sync(local_path=local_path)
        print("   ✓ 资料爬取完成")

        # 处理文档并创建向量化
        print("2. 处理文档并创建向量...")
        from utils.document_processor import DocumentProcessor
        processor = DocumentProcessor(embedding_model=embedding_model)

        # 处理各学科
        all_chunks = []
        base_path = os.path.dirname(os.path.abspath(__file__))
        exam_db_path = os.path.join(base_path, "exam_knowledge_db")

        for subject in subjects:
            subject_path = os.path.join(exam_db_path, subject)
            if os.path.exists(subject_path):
                chunks = processor.process_directory(subject_path, subject)
                all_chunks.extend(chunks)

        if all_chunks:
            print(f"   ✓ 文档分割完成，共 {len(all_chunks)} 个文档块")

            # 创建嵌入
            chunks_with_embeddings, embeddings = processor.create_embeddings(all_chunks)
            print(f"   ✓ 嵌入向量创建完成，共 {len(embeddings)} 个")

            # 创建向量数据库
            from database.create_db import create_db_info
            success = create_db_info(
                file_path=exam_db_path,
                embedding_model=embedding_model,
                subject="全部"
            )

            if success:
                print("✓ 知识库初始化完成")
                return True
            else:
                print("✗ 向量数据库创建失败")
                return False
        else:
            print("✗ 没有找到可处理的文档")
            return False

    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def optimize_system():
    """优化系统性能"""
    print("\n优化系统性能...")
    try:
        # 检查向量数据库
        vector_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exam_vector_db")
        if not os.path.exists(vector_db_path):
            print("   ⚠ 没有找到向量数据库，跳过优化")
            return True

        # 导入并运行优化器
        from optimizer import PerformanceOptimizer

        optimizer = PerformanceOptimizer()
        optimization_report = optimizer.auto_optimize()

        # 保存优化报告
        report_path = optimizer.save_optimization_report(optimization_report)
        print(f"✓ 系统优化完成，报告保存在: {report_path}")
        return True
    except Exception as e:
        print(f"✗ 优化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_gradio_app():
    """启动Gradio应用"""
    print("\n启动考研助手界面...")
    try:
        # 切换到serve目录
        serve_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve")
        os.chdir(serve_dir)

        # 启动Gradio应用
        subprocess.run([sys.executable, "run_gradio.py"])
        return True
    except Exception as e:
        print(f"✗ 启动失败: {e}")
        return False

def build_db_from_local():
    """从本地文件夹建库"""
    path = input("请输入本地文件夹路径（含PDF/TXT文件）: ").strip().strip('"')
    if not os.path.exists(path):
        print(f"路径不存在: {path}")
        return
    model_path = input("请输入m3e模型本地路径（直接回车使用在线模型）: ").strip().strip('"')
    from embedding.call_embedding import get_embedding
    from database.create_db import create_db_info
    # 预加载embedding以验证模型路径
    try:
        get_embedding("m3e", embedding_key=model_path if model_path else None)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    result = create_db_info(file_path=path, embedding_model="m3e",
                            persist_directory=None, subject="全部")
    print(result)


def main():
    """主函数"""
    print("=" * 60)
    print("南京理工大学控制工程考研助手")
    print("=" * 60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 检查依赖
    if not check_dependencies():
        return

    # 2. 初始化选项
    while True:
        print("\n请选择启动方式:")
        print("1. 首次启动（初始化+优化+运行）")
        print("2. 仅运行（已有知识库）")
        print("3. 初始化数据库")
        print("4. 系统优化")
        print("5. 从本地文件夹建库")
        print("6. 退出")

        choice = input("\n请输入选项(1-6): ")

        if choice == "1":
            print("\n执行首次启动流程...")
            local = input("请输入本地资料文件夹路径（直接回车跳过）: ").strip().strip('"') or None
            if initialize_database(local_path=local) and optimize_system():
                print("\n✓ 准备完成，即将启动界面...")
                time.sleep(2)
                start_gradio_app()
            else:
                print("\n✗ 首次启动失败")

        elif choice == "2":
            # 仅运行
            print("\n启动界面...")
            start_gradio_app()

        elif choice == "3":
            local = input("请输入本地资料文件夹路径（直接回车跳过）: ").strip().strip('"') or None
            if initialize_database(local_path=local):
                print("\n✓ 数据库初始化完成")

        elif choice == "4":
            # 系统优化
            if optimize_system():
                print("\n✓ 系统优化完成")

        elif choice == "5":
            build_db_from_local()

        elif choice == "6":
            print("退出程序")
            break

        else:
            print("无效选项，请重新输入")

if __name__ == "__main__":
    main()