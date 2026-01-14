"""语义记忆
结合向量检索和知识图谱的混合语义记忆，使用：
- Huggingface 中文预训练模型进行文本嵌入
- 向量相似度检索进行快速筛选
- 知识图谱进行实体关系推理
- 混合检索策略优化结果质量
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from app.memory.baseMemory import MemoryConfig, MemoryItem
from app.model.embedder import get_text_embedder
from app.utils.milvus_store import MilvusStore
from app.memory.storage.neo4j_store import Neo4jGraphStore

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Entity:
    def __init__(self, name: str, entity_id: str,
                 entity_type: str = "MISC",
                 description: str = "",
                 properties: Dict[str, Any] = None):
        self.entity_id = entity_id
        self.name = name
        self.entity_type = entity_type  # PERSON, ORG, PRODUCT, SKILL, CONCEPT等
        self.description = description
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.frequency = 1  # 出现频率

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "properties": self.properties,
            "frequency": self.frequency
        }

class Relation:
    """关系类"""
    def __init__(self,
                 from_entity: str,
                 to_entity: str,
                 relation_type: str,
                 strength: float = 1.0,
                 evidence: str = "",
                 properties: Dict[str, Any] = None):
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.relation_type = relation_type
        self.strength = strength
        self.evidence = evidence  # 支持该关系的原文本
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.frequency = 1  # 关系出现频率

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "evidence": self.evidence,
            "properties": self.properties,
            "frequency": self.frequency
        }


class SemanticMemory:
    """增强语义记忆实现
    特点:
    - 使用Huggingface中文预训练模型进行文本嵌入
    - 向量检索使用快速相似度匹配
    - 知识图谱存储实体和关系
    - 混合检索策略：向量+图+语义推理
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        
        # 嵌入模型
        self.embedding_model = None
        self._init_embedding_model()

        # 专业数据库存储
        self.graph_store: Optional[Neo4jGraphStore] = None
        # self.vector_store: Optional[MilvusStore] = None # 暂时注释，按需启用
        self._init_database()

        # 内存缓存
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        logger.info("增强语义记忆初始化完成 (使用 Neo4j 图数据库)")

    def _init_embedding_model(self):
        """初始化统一嵌入模型"""
        try:
            self.embedding_model = get_text_embedder()
            # 轻量健康检查
            if self.embedding_model:
                test_vec = self.embedding_model.embed_text("health_check")
                dim = len(test_vec)
                logger.info(f"嵌入模型初始化成功 (维度: {dim})")
        except Exception as e:
            logger.error(f"初始化统一嵌入模型失败: {e}")
            # 不抛出异常，允许降级运行

    def _init_database(self):
        """初始化专业数据库存储"""
        try:
            # 初始化图数据库
            try:
                self.graph_store = Neo4jGraphStore()
                if self.graph_store.health_check():
                    logger.info("Neo4j 图数据库连接成功")
                else:
                    logger.warning("Neo4j 连接检查失败，将在无图谱模式下运行")
                    self.graph_store = None
            except Exception as e:
                 logger.warning(f"Neo4j 初始化失败 ({e})，将在无图谱模式下运行")
                 self.graph_store = None

            # 向量数据库暂未完全启用，预留接口
            # self.vector_store = MilvusStore()

        except Exception as e:
            logger.error(f"数据库初始化错误: {e}")

    def add_entity(self, entity: Entity) -> bool:
        """添加实体到图数据库"""
        if not self.graph_store:
            return False
        
        return self.graph_store.add_entity(
            entity_id=entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            properties=entity.properties
        )

    def add_relation(self, relation: Relation) -> bool:
        """添加关系到图数据库"""
        if not self.graph_store:
            return False

        return self.graph_store.add_relationship(
            from_entity_id=relation.from_entity,
            to_entity_id=relation.to_entity,
            relationship_type=relation.relation_type,
            properties=relation.properties
        )

    def find_related(self, entity_id: str, depth: int = 1) -> List[Dict]:
        """查找相关实体"""
        if not self.graph_store:
             return []
        
        return self.graph_store.find_related_entities(
            entity_id=entity_id,
            max_depth=depth
        )
