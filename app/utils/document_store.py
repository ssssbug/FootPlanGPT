import time
import uuid

from sympy.polys.matrices.sdm import sdm_matmul
import json
import os.path
import threading
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, List
import psycopg2

class DocumentStore:
    """文档存储基类"""
    @abstractmethod
    def add_memory(self,
                   memory_id:str,
                   user_id:str,
                   content:str,
                   memory_type:str,
                   timestamp:int,
                   importance:float,
                   properties:Dict[str,Any]=None,):
        """添加记忆"""
        pass
    @abstractmethod
    def get_memory(self,memory_id:str)->Optional[Dict[str,Any]]:
        """获取单个记忆"""
        pass
    @abstractmethod
    def search_memories(self,
                        user_id:Optional[str]=None,
                        memory_type:Optional[str]=None,
                        start_time:Optional[str]=None,
                        end_time:Optional[str]=None,
                        importance_threshold:Optional[float]=None,
                        limit:int=10)->List[Dict[str,Any]]:
        """搜索记忆"""
        pass
    @abstractmethod
    def delete_memory(self,memory_id:str)->bool:
        """删除记忆"""
        pass


    @abstractmethod
    def get_database_stats(self)->Dict[str,Any]:
        """获取数据库统计信息"""
        pass

    def add_document(self,
                     document_id:str)->Optional[Dict[str,Any]]:
        """获取文档"""
        pass

class PostgreSQLDocumentStore(DocumentStore):
    """PostgreSQL文档存储实现"""

    _instances = {}#存储已创建的实例
    _initialized_dbs = set()#存储已初始化的数据库路径

    def __new__(cls,db_path:str="./memory.db"):
        """单例模式,同一路径只创建一个实例"""
        abs_path = os.path.abspath(db_path)
        if abs_path not in cls._instances:
            instance = super(PostgreSQLDocumentStore,cls).__new__(cls)
            cls._instances[abs_path]=instance
        return cls._instances[abs_path]
    def __init__(self,db_path:str="./memory.db"):
        #避免重复初始化
        if hasattr(self,"_initialized"):
            return
        self.db_path = db_path
        self.local = threading.local()

        #确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(db_path)),exist_ok=True)

        #初始化数据库(只初始化一次)
        abs_path = os.path.abspath(db_path)
        if abs_path not in self._initialized_dbs:
            self._init_database()#初始化数据库表
            self._initialized_dbs.add(abs_path)
            print(f"[OK] PostgreSQL 文档存储初始化完成:{abs_path}")
        self._initialized = True

    def _get_connection(self):
        """获取线程本地连接"""
        if not hasattr(self,"connection"):
            self.local.connection =psycopg2.connect(self.db_path,user="postgres")

        return self.local.connection

    def _init_database(self):
        """初始化数据库表"""
        conn = self._get_connection()
        cursor = conn.cursor()

        #创建用户表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users(
            id TEXT PRIMARY KEY,
            name TEXT,
            properties TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAM
            )
            """
        )
        #创建记忆表
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories(
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            importance REAL NOT NULL,
            properties TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
            """)


        #创建概念表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS concept(
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        propertis TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        """)
        #创建记忆-概念关联表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_concepts(
        memory_id TEXT NOT NULL,
        concept_id TEXT NOT NULL,
        relevance_score REAL DEFAULT 1.0
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(memory_id,concept_id),
        FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
        FOREIGN KEY(concept_id) REFERENCES concepts(id) ON DELETE CASCADE
        )
        """)
        #创建概念关系类
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS concept_relationships(
        from_concept_id TEXT NOT NULL,
        to_concept_id TEXT NOT NULL,
        relationship_type TEXT NOT NULL,
        strength REAL DEFAULT 1.0,
        properties TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY(from_concept_id,to_concept_id,relationship_type),
        FOREIGN KEY(from_concept_id) REFERENCES concepts(id) ON DELETE CASCADE,
        FOREIGN KEY(to_concept_id) REFERENCES concepts(id) ON DELETE CASCADE
        )
        """)

        #创建索引
        indexs = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)",
            "CREATE INDEX IF NOT EXISTS idx_memories_concepts_memory ON memory_concepts(memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_concepts_concept ON memory_concepts(concept_id)"
        ]

        for index_sql in indexs:
            cursor.execute(index_sql)
        conn.commit()
        print("[OK] PostgreSQL 数据库表和索引创建完成")


    def add_memory(self,
                   memory_id:str,
                   user_id:str,
                   content:str,
                   memory_type:str,
                   timestamp:int,
                   importance:float,
                   properties:Dict[str,Any]=None,):

        """添加记忆"""
        conn = self._get_connection()
        cursor=conn.cursor()

        #确保用户存在
        cursor.execute("""INSERT INTO users(id,name)
                       VALUES(%s,%s) ON CONFLICT (id) DO NOTHING""",(user_id,user_id))

        #插入记忆
        cursor.execute("""
        INSERT INTO memories
        (
            id,user_id,content,memory_type,timestamp,importance,properties,update_at
        )
            VALUES (
                   %s,%s,%s,%s,%s,%s,%s,CURRENT_TIMESTAMP
                   )
            ON CONFLICT (id)
            DO UPDATE SET
                user_id = EXCLUDED.user_id,
                content = EXCLUDED.content,
                memory_type = EXCLUDED.memory_type,
                timestamp = EXCLUDED.timestamp,
                importance = EXCLUDED.importance,
                properties = EXCLUDED.properties,
                update_at = CURRENT_TIMESTAMP;
            
        """,(
            memory_id,
            user_id,
            content,
            memory_type,
            timestamp,
            importance,
            properties
        ))
        conn.commit()
        return memory_id

    def get_memory(self,memory_id:str)->Optional[Dict[str,Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        """获取单个记忆"""
        cursor.execute("""
        SELECT id,user_id,content,memory_type,timestamp,importance,properties,created_at
        FROM memories
            WHERE id = %s
        
        """, (memory_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "memory_id":row["id"],
            "user_id":row["user_id"],
            "content":row["content"],
            "memory_type":row["memory_type"],
            "timestamp":row["timestamp"],
            "importance":row["importance"],
            "properties":json.loads(row["properties"]) if row["properties"] else{},
            "created_at":row["created_at"]
        }

    def search_memories(self,
                        user_id:Optional[str]=None,
                        memory_type:Optional[str]=None,
                        start_time:Optional[str]=None,
                        end_time:Optional[str]=None,
                        importance_threshold:Optional[float]=None,
                        limit:int=10) ->List[Dict[str,Any]]:

        """搜索记忆"""
        conn=self._get_connection()
        cursor = conn.cursor()

        #构建查询条件
        where_conditions=[]
        params = []
        if user_id:
            where_conditions.append("user_id = %s")
            params.append(user_id)
        if memory_type:
            where_conditions.append("memory_type = %s")
            params.append(memory_type)
        if start_time:
            where_conditions.append("timestamp >= %s")
            params.append(start_time)
        if end_time:
            where_conditions.append("timestamp <= %s")
            params.append(end_time)
        if importance_threshold:
            where_conditions.append("importance >= %s")
            params.append(importance_threshold)
        where_clause=""
        if where_conditions:
            where_clause="WHERE " + " AND ".join(where_conditions)
        
        #执行查询
        cursor.execute("""
        SELECT id,user_id,content,memory_type,timestamp,importance,properties,created_at
        FROM memories
        {where_clause}
        ORDER BY importance DESC,timestamp DESC
        LIMIT %s
        """,params+[limit])
        memories = []

        for row in cursor.fetchall():
            memories.append({
                "memory_id":row["id"],
                "user_id":row["user_id"],
                "content":row["content"],
                "memory_type":row["memory_type"],
                "timestamp":row["timestamp"],
                "importance":row["importance"],
                "properties":json.loads(row["properties"]) if row["properties"] else{},
                "created_at":row["created_at"]
            })

        return memories

    def update_memory(self,
                      memory_id:str,
                      content:str = None,
                      importance:float=None,
                      properties:Dict[str,Any]=None)->bool:
        
        """更新记忆"""
        conn=self._get_connection()
        cursor = conn.cursor()

        #构建更新字段
        update_fields=[]
        params = []

        if content is not None:
            update_fields.append("content = %s")
            params.append(content)
        if importance is not None:
            update_fields.append("importance = %s")
            params.append(importance)
        if properties is not None:
            update_fields.append("properties = %s")
            params.append(json.dumps(properties))
        if not update_fields:
            return False
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        params.append(memory_id)

        cursor.execute("""
        UPDATE memories
        SET{','.join(update_fields)}
        WHERE id = %s
        """,params)
        conn.commit()
        return cursor.rowcount > 0

    def delete_memory(self,memory_id:str)->bool:
        """删除记忆"""
        conn=self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""DELETE FROM memories WHERE id = %s""",(memory_id,))
        conn.commit()
        return cursor.rowcount > 0

    def get_database_stats(self)->Dict[str,Any]:
        """获取数据库统计信息"""
        conn=self._get_connection()
        cursor = conn.cursor()
        stats = {}

        #统计各表的记录数
        tables = ["users","memories","concepts","memory_concepts","concept_relationships"]
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) AS count FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()["count"]
        #统计记忆类型分布
        cursor.execute("""
        SELECT memory_type,COUNT(*) as count FROM memories GROUP BY memory_type
        """)
        memory_types={}
        for row in cursor.fetchall():
            memory_types[row["memory_type"]] = row["count"]
        stats["memory_types"] = memory_types

        #统计用户分布       
        cursor.execute(
            """
            SELECT user_id,COUNT(*) as count FROM memories GROUP BY user_id
            ORDER BY count DESC
            LIMIT 10
            """
        )
        top_users={}
        for row in cursor.fetchall():
            top_users[row["user_id"]] = row["count"]
        stats["top_users"] = top_users
        stats["store_type"]="postgresql"
        stats["db_path"]=self.db_path
        return stats

    def add_document(self,content:str,metadata:Dict[str,Any]=None)->str:
        """添加文档"""
        doc_id  = str(uuid.uuid4())
        user_id = metadata.get("user_id","system") if metadata else "system"

        return self.add_memory(
            memory_id = doc_id,
            user_id = user_id,
            content = content,
            memory_type= "document",
            timestamp = int(time.time()),
            importance = 0.5,
            properties = metadata or {}
        )

    def get_document(self,document_id:str)->Optional[Dict[str,Any]]:
        """获取文档"""
        return self.get_memory(document_id)
    
    def close(self):
        if hasattr(self.local,'connection'):
            self.local.connection.close()
            delattr(self.local,'connection')
            print("[OK] PostgreSQL 连接已关闭")



























