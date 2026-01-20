#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识文档入库脚本

使用方法:
    python ingest_documents.py /path/to/doc1.pdf /path/to/doc2.md ...
    
或者直接修改下面的 DOCUMENT_PATHS 列表，然后运行:
    python ingest_documents.py
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# ========================================
# 配置区域 - 在这里填入你的文档路径
# ========================================

DOCUMENT_PATHS = [
    # 示例路径，请替换为你的实际文档路径
    # "/Users/hy/Documents/营养学资料.pdf",
    # "/Users/hy/Documents/食谱大全.md",
    # "/Users/hy/Documents/高血压饮食指南.docx",
]

# RAG 配置
RAG_NAMESPACE = "food_kb"           # 知识库命名空间
COLLECTION_NAME = "food_recipe_kb"  # 向量集合名称
CHUNK_SIZE = 800                    # 分块大小 (tokens)
CHUNK_OVERLAP = 100                 # 分块重叠 (tokens)

# ========================================
# 主程序
# ========================================

def main():
    from services.rag_pipeline import create_rag_pipeline
    
    # 获取文档路径 (命令行参数优先，否则使用配置的列表)
    if len(sys.argv) > 1:
        doc_paths = sys.argv[1:]
    else:
        doc_paths = DOCUMENT_PATHS
    
    if not doc_paths:
        print("=" * 60)
        print("错误: 未指定任何文档路径!")
        print()
        print("使用方法 1: 命令行参数")
        print("    python ingest_documents.py /path/to/doc1.pdf /path/to/doc2.md")
        print()
        print("使用方法 2: 编辑脚本中的 DOCUMENT_PATHS 列表")
        print("    打开此文件，在 DOCUMENT_PATHS 中添加文档路径")
        print("=" * 60)
        sys.exit(1)
    
    # 验证文件存在
    valid_paths = []
    for p in doc_paths:
        if os.path.exists(p):
            valid_paths.append(p)
            print(f"[✓] 找到文档: {p}")
        else:
            print(f"[✗] 文档不存在: {p}")
    
    if not valid_paths:
        print("\n错误: 没有有效的文档路径!")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print(f"开始入库 {len(valid_paths)} 个文档")
    print(f"命名空间: {RAG_NAMESPACE}")
    print(f"分块大小: {CHUNK_SIZE} tokens, 重叠: {CHUNK_OVERLAP} tokens")
    print("=" * 60)
    print()
    
    # 初始化 RAG Pipeline
    try:
        pipeline = create_rag_pipeline(
            collection_name=COLLECTION_NAME,
            rag_namespace=RAG_NAMESPACE,
            neo4j_enabled=True
        )
        print("[✓] RAG Pipeline 初始化成功")
    except Exception as e:
        print(f"[✗] RAG Pipeline 初始化失败: {e}")
        sys.exit(1)
    
    # 入库文档
    try:
        chunk_count = pipeline["add_documents"](
            file_paths=valid_paths,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        print()
        print("=" * 60)
        print(f"[成功] 入库完成!")
        print(f"  - 处理文档数: {len(valid_paths)}")
        print(f"  - 生成分块数: {chunk_count}")
        print("=" * 60)
        
        # 显示统计信息
        stats = pipeline["get_stats"]()
        print()
        print("知识库统计:")
        print(f"  - 向量数据库: {stats}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[✗] 入库失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
