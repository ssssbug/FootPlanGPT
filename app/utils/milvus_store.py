"""
Milvus向量数据库存储实现 - 工具函数扩展
"""

import os
import threading
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pymilvus import MilvusClient, DataType

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusVectorStore:
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "default_collection",
        vector_size: int = 384,
        distance: str = "COSINE",
        timeout: int = 30,
        **kwargs
    ):
        """
        初始化MilvusVectorStore
        :param url: Milvus服务地址或本地DB路径(如 ./milvus_demo.db)
        :param api_key: 认证Token (对于Zilliz Cloud或启用认证的Milvus)
        :param collection_name: 集合名称
        :param vector_size: 向量维度
        :param distance: 距离度量 (COSINE, L2, IP)
        :param timeout: 超时时间
        """
        self.url = url or "./milvus_demo.db"
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout
        
        # 转换距离度量
        self.metric_type = distance.upper() if distance else "COSINE"
        if self.metric_type == "DOT": # 兼容性处理
            self.metric_type = "IP" # Inner Product
        elif self.metric_type == "EUCLIDEAN":
            self.metric_type = "L2"
        
        self.client = None
        self._initialize_client()
        self._init_collection()

    def _initialize_client(self):
        """初始化Milvus客户端"""
        try:
            # 如果是本地文件模式
            if self.url.endswith('.db'):
                # 确保目录存在
                db_dir = os.path.dirname(os.path.abspath(self.url))
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir, exist_ok=True)
            
            self.client = MilvusClient(
                uri=self.url,
                token=self.api_key,
                timeout=self.timeout
            )
            logger.info(f"成功连接到Milvus: {self.url}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise e

    def _init_collection(self):
        """初始化集合"""
        try:
            if not self.client.has_collection(self.collection_name):
                schema = self.client.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                    description=f"Vector store for {self.collection_name}"
                )
                
                schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
                schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.vector_size)
                schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
                
                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    index_type="FLAT",
                    metric_type=self.metric_type,
                    params={}
                )
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=index_params
                )
                logger.info(f"创建集合: {self.collection_name}, 维度: {self.vector_size}, Metric: {self.metric_type}")
            else:
                logger.info(f"使用已有集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"初始化集合失败: {e}")
            raise e

    def add_vectors(
        self, 
        vectors: List[List[float]], 
        ids: List[str], 
        texts: Optional[List[str]] = None, 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """添加向量数据到Milvus"""
        if not vectors or not ids:
            logger.warning("向量列表或ID列表为空")
            return False
        
        if len(vectors) != len(ids):
            raise ValueError(f"向量数量({len(vectors)})与ID数量({len(ids)})不匹配")
        
        n_meta = len(metadatas) if metadatas else 0
        n_texts = len(texts) if texts else 0
        logger.info(f"【Milvus】添加向量数据: n_vectors={len(vectors)}, n_meta={n_meta}, n_texts={n_texts}, collection={self.collection_name}")

        data = []
        for i, (vector, vec_id) in enumerate(zip(vectors, ids)):
            if len(vector) != self.vector_size:
                logger.warning(f"向量[{i}]维度不匹配: {len(vector)} != {self.vector_size}, 跳过")
                continue
            
            safe_id = str(vec_id)
            item = {"id": safe_id, "vector": vector}
            
            if texts and i < len(texts):
                item["text"] = texts[i] if texts[i] else ""
            else:
                item["text"] = ""
            
            if metadatas and i < len(metadatas) and metadatas[i]:
                metadata = metadatas[i].copy()
                metadata["timestamp"] = int(datetime.now().timestamp())
                metadata["added_at"] = int(datetime.now().timestamp())
                
                if "external" in metadata and not isinstance(metadata.get("external"), bool):
                    val = metadata.get("external")
                    metadata["external"] = str(val).lower() in ("1", "true", "yes")
                
                for key, value in metadata.items():
                    if key not in ["id", "vector", "text"]:
                        item[key] = value
            
            data.append(item)
        
        if not data:
            logger.warning("没有有效的向量数据可插入")
            return False
        
        try:
            logger.info(f"[Milvus] 开始插入 {len(data)} 条数据")
            res = self.client.insert(collection_name=self.collection_name, data=data)
            logger.info(f"[Milvus] 插入完成，成功添加 {len(data)} 个向量到 {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            raise e

    def search_similar(
        self, 
        query_vector: List[float], 
        top_k: int = 5, 
        filter_expr: str = "",
        output_fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """向量相似度搜索"""
        logger.info(f"[Milvus Search] 开始向量搜索")
        logger.info(f"  - 查询向量维度: {len(query_vector)}")
        logger.info(f"  - 距离度量: {self.metric_type}")
        logger.info(f"  - Top-K: {top_k}")
        logger.info(f"  - 过滤条件: {filter_expr if filter_expr else '无'}")
        
        if output_fields is None:
            output_fields = ["id", "text", "*"]
            
        search_params = {"metric_type": self.metric_type, "params": {}}
        
        try:
            logger.info(f"[Milvus Search] 执行搜索...")
            res = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                filter=filter_expr,
                limit=top_k,
                output_fields=output_fields,
                search_params=search_params
            )
            
            logger.info(f"[Milvus Search] 搜索完成")
            results = []
            if res and len(res) > 0:
                logger.info(f"[Milvus Search] 找到 {len(res[0])} 个相似结果:")
                
                for idx, hit in enumerate(res[0]):
                    item = hit['entity']
                    distance = hit['distance']
                    item['score'] = distance
                    
                    if self.metric_type == "COSINE":
                        similarity_pct = (distance + 1) / 2 * 100
                        logger.info(f"  [{idx+1}] ID: {str(item.get('id', 'N/A'))[:20]}, "
                                  f"余弦相似度: {distance:.4f}, "
                                  f"相似度百分比: {similarity_pct:.2f}%")
                    elif self.metric_type == "L2":
                        logger.info(f"  [{idx+1}] ID: {str(item.get('id', 'N/A'))[:20]}, "
                                  f"L2距离: {distance:.4f} (越小越相似)")
                    elif self.metric_type == "IP":
                        logger.info(f"  [{idx+1}] ID: {str(item.get('id', 'N/A'))[:20]}, "
                                  f"内积: {distance:.4f} (越大越相似)")
                    else:
                        logger.info(f"  [{idx+1}] ID: {str(item.get('id', 'N/A'))[:20]}, "
                                  f"距离分数: {distance:.4f}")
                    
                    results.append(item)
            else:
                logger.warning("[Milvus Search] 未找到匹配结果")
                    
            return results
        except Exception as e:
            logger.error(f"[Milvus Search] 搜索失败: {e}")
            raise e

    def delete(self, ids: List[str]) -> bool:
        """删除向量"""
        if not ids:
            return False
            
        try:
            res = self.client.delete(collection_name=self.collection_name, ids=ids)
            cnt = 0
            if isinstance(res, list):
                for r in res:
                    if isinstance(r, dict):
                        cnt += r.get("delete_count", 0)
                    elif hasattr(r, "delete_count"):
                        cnt += r.delete_count
            elif isinstance(res, dict):
                cnt = res.get("delete_count", 0)
            elif hasattr(res, "delete_count"):
                 cnt = res.delete_count
                 
            return cnt > 0
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False

    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        """通过ID获取向量"""
        if not ids:
            return []
        try:
            res = self.client.get(collection_name=self.collection_name, ids=ids)
            return res
        except Exception as e:
            logger.error(f"获取向量失败: {e}")
            return []

    def count(self) -> int:
        """获取向量数量"""
        try:
             res = self.client.query(
                 collection_name=self.collection_name,
                 filter="", 
                 output_fields=["count(*)"]
             )
             if res:
                 return res[0].get("count(*)")
        except Exception:
            try:
                stats = self.client.get_collection_stats(self.collection_name)
                return stats.get("row_count", 0)
            except Exception:
                pass
        return 0

    def clear_collection(self) -> bool:
        """清空集合中的所有数据"""
        try:
            logger.warning(f"[Milvus] 准备清空集合: {self.collection_name}")
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                logger.info(f"[Milvus] 已删除集合: {self.collection_name}")
            
            self._init_collection()
            logger.info(f"[Milvus] 集合已清空并重新创建")
            return True
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False

    def delete_by_filter(self, filter_expr: str) -> bool:
        """根据过滤条件删除向量"""
        if not filter_expr:
            logger.warning("过滤表达式为空,拒绝删除")
            return False
        
        try:
            logger.info(f"[Milvus] 按条件删除: {filter_expr}")
            res = self.client.delete(collection_name=self.collection_name, filter=filter_expr)
            logger.info(f"[Milvus] 删除完成")
            return True
        except Exception as e:
            logger.error(f"按条件删除失败: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合详细信息"""
        try:
            info = {
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
                "metric_type": self.metric_type,
                "exists": self.client.has_collection(self.collection_name)
            }
            
            if info["exists"]:
                desc = self.client.describe_collection(self.collection_name)
                info["schema"] = desc
                
            return info
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = {
                "collection_name": self.collection_name,
                "total_count": self.count(),
                "vector_dimension": self.vector_size,
                "metric_type": self.metric_type,
                "db_path": self.url
            }
            
            try:
                coll_stats = self.client.get_collection_stats(self.collection_name)
                stats["collection_stats"] = coll_stats
            except Exception:
                pass
                
            logger.info(f"[Milvus Stats] {stats}")
            return stats
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health = {
            "status": "unknown",
            "collection_exists": False,
            "can_query": False,
            "can_insert": False,
            "error": None
        }
        
        try:
            health["collection_exists"] = self.client.has_collection(self.collection_name)
            
            if not health["collection_exists"]:
                health["status"] = "error"
                health["error"] = f"集合 {self.collection_name} 不存在"
                return health
            
            try:
                count = self.count()
                health["can_query"] = True
                health["total_vectors"] = count
            except Exception as e:
                health["error"] = f"查询失败: {e}"
                
            try:
                test_id = f"health_check_{int(datetime.now().timestamp())}"
                test_vector = [0.0] * self.vector_size
                
                self.client.insert(
                    collection_name=self.collection_name,
                    data=[{"id": test_id, "vector": test_vector, "text": "health_check"}]
                )
                health["can_insert"] = True
                
                self.client.delete(collection_name=self.collection_name, ids=[test_id])
            except Exception as e:
                health["error"] = f"插入测试失败: {e}"
            
            if health["can_query"] and health["can_insert"]:
                health["status"] = "healthy"
            elif health["can_query"]:
                health["status"] = "degraded"
            else:
                health["status"] = "error"
                
            logger.info(f"[Milvus Health] {health['status']}")
            return health
            
        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
            logger.error(f"健康检查失败: {e}")
            return health


class MilvusConnectionManager:
    """Milvus连接管理器 (单例工厂)"""
    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        url: str = "./milvus_demo.db",
        collection_name: str = "default",
        metric_type: str = "L2",
        vector_size: int = 384,
        **kwargs
    ) -> MilvusVectorStore:
        """获取或创建Milvus实例"""
        key = f"{url}_{collection_name}"
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = MilvusVectorStore(
                    url=url,
                    collection_name=collection_name,
                    vector_size=vector_size,
                    distance=metric_type,
                    **kwargs
                )
            return cls._instances[key]