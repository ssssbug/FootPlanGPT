
import os
import uuid
import time
import sys
from datetime import datetime

# 确保项目路径在 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.utils.document_store import MilvusDocumentStore, PostgreSQLDocumentStore
from app.utils.milvus_store import MilvusVectorStore, MilvusConnectionManager
from app.model.embedder import get_text_embedder

def print_banner(text):
    print("\n" + "="*60)
    print(f" {text.center(58)} ")
    print("="*60 + "\n")

def test_milvus_document_store():
    print_banner("Testing MilvusDocumentStore")
    try:
        # 1. 初始化 (使用独立的测试库)
        db_path = "./test_milvus_document.db"
        if os.path.exists(db_path):
            os.remove(db_path)
            
        store = MilvusDocumentStore(db_path=db_path)
        print(f"[OK] 成功初始化 MilvusDocumentStore: {db_path}")

        # 2. 添加记忆
        mem_id = str(uuid.uuid4())
        user_id = "test_user_001"
        content = "这是一个关于健康饮食的测试记忆。"
        
        print(f"[INFO] 正在插入数据: {content}")
        store.add_memory(
            memory_id=mem_id,
            user_id=user_id,
            content=content,
            memory_type="test",
            timestamp=int(time.time()),
            importance=0.8,
            properties={"category": "health"}
        )
        print(f"[OK] 数据插入成功, ID: {mem_id}")

        # 3. 检索单个记忆
        retrieved = store.get_memory(mem_id)
        if retrieved:
            print(f"[OK] 检索到记忆: {retrieved['content']}")
        else:
            print("[FAIL] 未检索到记忆")

        # 4. 搜索记忆 (标量过滤)
        print("[INFO] 正在按用户ID搜索...")
        results = store.search_memories(user_id=user_id, limit=5)
        print(f"[OK] 搜索结果数量: {len(results)}")
        for r in results:
            print(f"  - [{r['memory_id']}] {r['content']} (重要度: {r['importance']})")

        # 5. 更新记忆
        print("[INFO] 正在更新内容...")
        store.update_memory(mem_id, content="这是更新后的健康饮食测试记忆。", importance=0.9)
        updated = store.get_memory(mem_id)
        print(f"[OK] 更新后内容: {updated['content']}, 重要度: {updated['importance']}")

        # 6. 数据库统计
        stats = store.get_database_stats()
        print(f"[OK] 数据库统计: {stats}")

        # 7. 删除
        print("[INFO] 正在删除数据...")
        store.delete_memory(mem_id)
        if not store.get_memory(mem_id):
            print("[OK] 数据已成功删除")
        
        store.close()
        
    except Exception as e:
        print(f"[ERROR] MilvusDocumentStore 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_milvus_vector_store():
    print_banner("Testing MilvusVectorStore (Pure Vector Search)")
    try:
        # 1. 初始化
        db_path = "./test_milvus_vector.db"
        if os.path.exists(db_path):
            os.remove(db_path)
            
        embedder = get_text_embedder()
        dim = embedder.dimension
        
        vector_store = MilvusVectorStore(
            url=db_path,
            collection_name="test_vector_collection",
            vector_size=dim
        )
        print(f"[OK] 成功初始化 MilvusVectorStore, 维度: {dim}")

        # 2. 准备数据
        docs = [
            "红烧肉的做法非常讲究，需要慢火炖煮。",
            "清蒸鱼要保证鱼肉的鲜嫩，火候很关键。",
            "西红柿炒鸡蛋是国民菜肴，简单又营养。",
            "运动后应该补充蛋白质和水分。"
        ]
        
        vectors = [embedder.encode(d) if not hasattr(embedder.encode(d), 'tolist') else embedder.encode(d).tolist() for d in docs]
        ids = [str(uuid.uuid4()) for _ in range(len(docs))]
        metadatas = [{"type": "recipe"} if "做" in d or "菜" in d else {"type": "body"} for d in docs]

        # 3. 插入向量
        print(f"[INFO] 正在插入 {len(docs)} 条向量数据...")
        vector_store.add_vectors(vectors=vectors, ids=ids, texts=docs, metadatas=metadatas)
        print(f"[OK] 插入成功, 当前总量: {vector_store.count()}")

        # 4. 向量相似度搜索
        query_text = "怎么做肉菜？"
        print(f"[INFO] 正在执行语义搜索: '{query_text}'")
        query_vec = embedder.encode(query_text)
        if hasattr(query_vec, 'tolist'):
            query_vec = query_vec.tolist()
            
        results = vector_store.search_similar(query_vec, top_k=2)
        print(f"[OK] 语义搜索结果 (Top 2):")
        for i, res in enumerate(results):
            score = res.get('score', 0)
            print(f"  {i+1}. [相似度: {score:.4f}] 内容: {res.get('text')}")

        # 5. 健康检查与统计
        health = vector_store.health_check()
        print(f"[OK] 健康检查: {health['status']}")
        
        stats = vector_store.get_stats()
        print(f"[OK] 统计信息: {stats}")

        # 6. 清理
        vector_store.delete(ids=[ids[0]])
        print(f"[OK] 删除一条数据后总量: {vector_store.count()}")
        
    except Exception as e:
        print(f"[ERROR] MilvusVectorStore 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_postgresql_document_store():
    print_banner("Testing PostgreSQLDocumentStore")
    print("[INFO] 尝试连接 PostgreSQL...")
    try:
        # 这里默认尝试连接本地 postgres，如果没装会报错
        store = PostgreSQLDocumentStore()
        print("[OK] 连接成功 (或已存在实例)")
        # 如果需要进一步测试，可以取消下面注释
        # 但考虑到环境可能没有 PG，这里只做连接尝试
        stats = store.get_database_stats()
        print(f"[OK] 数据库统计: {stats}")
    except Exception as e:
        print(f"[SKIP] PostgreSQL 测试跳过: 请确保已安装 psycopg2 且本地 PostgreSQL 服务已启动。")
        print(f"       错误详情: {e}")

if __name__ == "__main__":
    test_milvus_document_store()
    test_milvus_vector_store()
    test_postgresql_document_store()
    
    print_banner("All Tests Completed")
