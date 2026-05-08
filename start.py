#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
考研助手启动脚本
"""
import os
import sys
import argparse

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser(description="考研助手服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    args = parser.parse_args()

    print("=" * 50)
    print("考研助手服务启动中...")
    print(f"API: http://{args.host}:{args.port}/api")
    print(f"前端: http://{args.host}:{args.port}/")
    print(f"项目根目录: {project_root}")
    print("=" * 50)

    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

if __name__ == "__main__":
    main()
