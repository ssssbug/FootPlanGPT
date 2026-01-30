from dotenv import load_dotenv
from _pickle import load
from typing import Optional
from typing import List, Dict, Any
from datetime import datetime
import logging
import os
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None


logger = logging.getLogger(__name__)

class Neo4jGraphStore:
    """Neo4j图数据库存储实现"""

    def __init__(self, uri: str = Optional[str]=None,
                 username: str = Optional[str]=None,
                 password: str = Optional[str]=None,
                 database_name: str = Optional[str]=None,
                 max_connection_pool_size: int = 3600,
                 max_connection_lifetime: int = 50,
                 connection_acquisition_timeout: int = 60,
                 **kwargs):
        """
        初始化Neo4j图存储(支持云API)
        Args:
            uri: Neo4j连接URI (本地: bolt://localhost:7687, 云: neo4j+s://xxx.database.neo4j.io)
            username: 用户名
            password: 密码
            database_name: 数据库名称
            max_connection_pool_size: 最大连接池大小
            max_connection_lifetime: 最大连接生命周期
            connection_acquisition_timeout: 连接获取超时
        """

        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j图数据库存储需要安装 pip install neo4j")
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database_name

        # 初始化驱动
        self.driver = None
        self._initialize_driver(
            max_connection_lifetime=max_connection_lifetime,
            connection_acquisition_timeout=connection_acquisition_timeout,
            max_connection_pool_size=max_connection_pool_size
        )
        # 创建索引
        self._create_indexes()

    def _initialize_driver(self, **config):
        """初始化Neo4j驱动"""
        try:
            self.driver = GraphDatabase.driver(self.uri,
                                               auth=(self.username, self.password),
                                               **config)
            # 验证连接
            self.driver.verify_connectivity()
            # 检查是否是云服务
            if "neo4j.io" in self.uri or "aura" in self.uri.lower():
                logger.info(f"成功连接到Neo4j云服务: {self.uri}")

            else:
                logger.info(f"成功连接到本地Neo4j数据库: {self.uri}")
            
        except AuthError as e:
            logger.error(f"Neo4j认证失败: {e}")
            logger.info("请检查Neo4j的用户名和密码")
            raise

        except ServiceUnavailable as e:
            logger.error(f"Neo4j服务不可用: {e}")
            if "localhost" in self.uri:
                logger.info("本地连接失败,可以考虑使用云服务")
                logger.info("或启动本地服务: docker run -p 7474:7474 -p 7687:7687 neo4j:5.14")
            else:
                logger.info("请检查URL和网络连接")
            raise
        except Exception as e:
            logger.error(f"初始化Neo4j驱动失败: {e}")
            raise

    def _create_indexes(self):
        """创建必要的索引提高查询性能"""
        indexes = [
            # 实体索引
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",

            # 记忆索引
            "CREATE INDEX memory_id_index IF NOT EXISTS FOR (m:Memory) ON (m.id)",
            "CREATE INDEX memory_type_index IF NOT EXISTS FOR (m:Memory) ON (m.type)",
            "CREATE INDEX memory_timestamp_index IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)"
        ]
        with self.driver.session(database=self.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                except Exception as e:
                    logger.debug(f"索引创建跳过（可能已存在）: {e}")
        logger.info("Neo4j索引创建完成")

    def add_entity(self, entity_id: str, name: str, entity_type: str, properties: Dict[str, Any] = None) -> bool:
        """添加实体
        Args:
            entity_id: 实体ID
            name: 实体名称
            entity_type: 实体类型
            properties: 实体属性
        Returns:
            bool: 是否成功
        """

        try:
            props = properties or {}
            props.update({
                "id": entity_id,
                "name": name,
                "type": entity_type,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            query = """
            MERGE (e:Entity{id:$entity_id})
            SET e+=$properties
            RETURN e
            """
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id, properties=props)
                record = result.single()

                if record:
                    logger.info(f"添加实体: {name}({entity_type})")
                    return True
                return False
        except Exception as e:
            logger.error(f"添加实体失败: {e}")
            return False
    
    def add_relationship(
        self,
        from_entity_id: str,
        to_entity_id: str,
        relationship_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        添加实体间关系
        Args:
            from_entity_id: 源实体ID
            to_entity_id: 目标实体ID
            relationship_type: 关系类型
            properties: 关系属性
        Returns:
            bool: 是否成功
        """
        try:
            props = properties or {}
            props.update({
                "type": relationship_type,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            })
            query = f"""
            MATCH (from:Entity{{id:$from_id}})
            MATCH (to:Entity{{id:$to_id}})
            MERGE (from)-[r:{relationship_type}]->(to)
            SET r+=$properties
            RETURN r
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    from_id=from_entity_id,
                    to_id=to_entity_id,
                    properties=props
                )

                record = result.single()

                if record:
                    logger.debug(f"添加关系: {from_entity_id}-{relationship_type}->{to_entity_id}")
                    return True
                return False

        except Exception as e:
            logger.error(f"添加关系失败: {e}")
            return False

    def find_related_entities(
        self,
        entity_id: str,
        relationship_types: List[str] = None,
        max_depth: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        查找相关实体
        Args:
            entity_id: 起始实体ID
            relationship_types: 关系类型过滤
            max_depth: 最大搜索深度
            limit: 结果限制
        Returns:
            List[Dict]: 相关实体列表
        """
        try:
            # 构建关系类型过滤
            rel_filter = ""
            if relationship_types:
                rel_type = "|".join(relationship_types)
                rel_filter = f":{rel_type}"

            query = f"""
            MATCH path=(start:Entity{{id:$entity_id}})-[r{rel_filter}*1..{max_depth}]-(related:Entity)
            WHERE start.id <> related.id
            RETURN DISTINCT related,
                            length(path) as distance,
                            [rel in relationships(path) | type(rel)] as relationship_path
            ORDER BY distance, related.name
            LIMIT $limit
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id, limit=limit)

                entities = []
                for record in result:
                    entity_data = dict(record["related"])
                    entity_data["distance"] = record["distance"]
                    entity_data["relationship_path"] = record["relationship_path"]
                    entities.append(entity_data)

                logger.debug(f"找到 {len(entities)} 个相关实体")

                return entities

        except Exception as e:
            logger.error(f"查找相关实体失败: {e}")
            return []

    def search_entities_by_name(self, name_pattern: str, entity_types: List[str] = None, limit: int = 20) -> List[Dict[str, Any]]:

        """按名称搜索实体
        Args:
            name_pattern: 名称模式(支持部分匹配)
            entity_types: 实体类型过滤
            limit: 结果限制
        Returns:
            List[Dict]: 匹配的实体列表
        """
        try:
            # 构建类型过滤
            type_filter = ""
            params = {"pattern": f".*{name_pattern}.*", "limit": limit}

            if entity_types:
                type_filter = "AND e.type IN $types"
                params["types"] = entity_types

            query = f"""
            MATCH (e:Entity)
            WHERE e.name =~ $pattern {type_filter}
            RETURN e
            ORDER BY e.name
            LIMIT $limit
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(query, **params)
                entities = []
                for record in result:
                    entity_data = dict(record["e"])
                    entities.append(entity_data)

                logger.debug(f"按名称搜索到 {len(entities)} 个实体")
                return entities

        except Exception as e:
            logger.error(f"按名称搜索实体失败: {e}")
            return []

    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:

        """
        获取实体的所有关系
        Args:
            entity_id: 实体ID
        Returns:
            List[Dict]: 关系列表
        """

        try:
            query = """
            MATCH (e:Entity{id:$entity_id})-[r]-(other:Entity)
            RETURN r, other,
                    CASE WHEN startNode(r).id = $entity_id THEN 'outgoing' ELSE 'incoming' END as direction
            """
            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id)

                relationships_list = []
                for record in result:
                    rel_data = dict(record["r"])
                    other_data = dict(record["other"])

                    rel_item = {
                        "relationship": rel_data,
                        "other_entity": other_data,
                        "direction": record["direction"]
                    }
                    relationships_list.append(rel_item)
            return relationships_list

        except Exception as e:
            logger.error(f"获取实体关系失败: {e}")
            return []

    def delete_entity(self, entity_id: str) -> bool:
        """
        删除实体及其所有关系
        Args:
            entity_id: 实体ID
        Returns:
            bool: 是否成功
        """
        try:
            query = """
            MATCH (e:Entity {id:$entity_id})
            DETACH DELETE e
            """

            with self.driver.session(database=self.database) as session:
                result = session.run(query, entity_id=entity_id)
                summary = result.consume()

                deleted_count = summary.counters.nodes_deleted
                logger.info(f"删除实体: {entity_id}(删除 {deleted_count} 个节点)")
                return deleted_count > 0

        except Exception as e:
            logger.error(f"删除实体失败: {e}")
            return False

    def clear_all(self) -> bool:
        """清空所有数据
        Returns:
            bool: 是否成功
        """
        try:
            query = "MATCH (n) DETACH DELETE n"

            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                summary = result.consume()

                deleted_nodes = summary.counters.nodes_deleted
                deleted_relationships = summary.counters.relationships_deleted

                logger.info(f"清空Neo4j数据库：删除 {deleted_nodes} 个节点，{deleted_relationships} 个关系")
                return True

        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        获取图数据库统计信息
        Returns:
            Dict: 统计信息
        """
        try:
            queries = {
                "total_nodes": "MATCH (n) RETURN count(n) as count",
                "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
                "entity_nodes": "MATCH (n:Entity) RETURN count(n) as count",
                "memory_nodes": "MATCH (n:Memory) RETURN count(n) as count",
            }
            stats = {}
            with self.driver.session(database=self.database) as session:
                for key, query in queries.items():
                    result = session.run(query)
                    record = result.single()
                    stats[key] = record["count"] if record else 0

            return stats
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    def health_check(self) -> bool:
        """
        健康检查
        Returns:
            bool: 服务是否健康
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as health")
                record = result.single()
                return record["health"] == 1
        
        except Exception as e:
            logger.error(f"Neo4j健康检查失败: {e}")
            return False
    
    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass


class Neo4jConnectionManager:
    """Singleton manager for Neo4j connection"""
    _instance = None
    _lock = None
    
    @classmethod
    def get_instance(cls, 
                     uri: Optional[str] = None, 
                     username: Optional[str] = None, 
                     password: Optional[str] = None,
                     database_name: Optional[str] = None) -> 'Neo4jGraphStore':
        import threading
        if cls._lock is None:
            cls._lock = threading.Lock()
            
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    import os
            
                    final_uri = uri or os.getenv("NEO4J_URI", "")
                    final_user = username or os.getenv("NEO4J_USER", "")
                    final_pwd = password or os.getenv("NEO4J_PASSWORD", "")
                    final_db = database_name or os.getenv("NEO4J_DATABASE", "")
                    
                    try:
                        cls._instance = Neo4jGraphStore(
                            uri=final_uri,
                            username=final_user,
                            password=final_pwd,
                            database_name=final_db
                        )
                        logger.info("Neo4jConnectionManager initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize Neo4jConnectionManager: {e}")
                        return None
        return cls._instance

# load_dotenv()
# if __name__ == "__main__":
#     # print("CWD:", os.getcwd())
#     # final_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
#     # print(final_uri)
#     manager = Neo4jConnectionManager.get_instance()
#     if manager:
#         print("Neo4jConnectionManager initialized successfully")
#     manager.health_check()
#     result=manager.add_entity(12,"test","test")
#     if result:
#         print("Entity added successfully")