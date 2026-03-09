"""
情景记忆实现
提供:
- 具体交互事件存储
- 时间序列组织
- 上下文丰富的记忆
- 模式识别能力
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

from app.memory.baseMemory import MemoryConfig, BaseMemory, MemoryItem
from app.model.embedder import get_text_embedder
from app.utils.milvus_store import MilvusConnectionManager

logger = logging.getLogger(__name__)


class Episode:
    """情景记忆中的单个场景"""
    def __init__(self,
                 episode_id: str,
                 user_id: str,
                 session_id: str,
                 timestamp: datetime,
                 content: str,
                 context: Dict[str, Any],
                 outcome: Optional[str] = None,
                 importance: float = 0.5
                 ):
        self.episode_id = episode_id
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.content = content
        self.context = context
        self.outcome = outcome
        self.importance = importance


class EpisodicMemory(BaseMemory):
    """情景记忆实现
    特点:
    - 存储具体的交互事件
    - 包含丰富的上下文信息
    - 按时间序列组织
    - 支持模式识别和回溯
    - 使用 MilvusLite 作为本地持久化存储
    """
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)

        # 本地缓存(内存)
        self.episodes: List[Episode] = []
        self.sessions: Dict[str, List[str]] = {}  # session_id -> episode_ids

        # 模式识别缓存
        self.patterns_cache = {}
        self.last_pattern_analysis = None

        # 统一嵌入模型(多语言，默认384维)
        self.embedder = get_text_embedder()

        # 向量存储(Milvus Lite / Remote)
        milvus_url = os.getenv("MILVUS_URL", "./episodic_memory.db")
        milvus_api_key = os.getenv("MILVUS_API_KEY")
        try:
            self.vector_store = MilvusConnectionManager.get_instance(
                url=milvus_url,
                collection_name="episodes",
                metric_type="COSINE",
                vector_size=384,
            )
        except Exception as e:
            logger.warning(f"[EpisodicMemory] 向量库初始化失败: {e}，将仅使用内存缓存")
            self.vector_store = None

        # 重量级持久化存储（可选，使用 MilvusDocumentStore 本地 lite 模式）
        try:
            from app.utils.document_store import MilvusDocumentStore
            doc_db_path = os.path.join(
                self.config.storage_path if hasattr(self.config, "storage_path") else ".memory_data",
                "episodic_doc.db"
            )
            os.makedirs(os.path.dirname(os.path.abspath(doc_db_path)), exist_ok=True)
            self.doc_store = MilvusDocumentStore(db_path=doc_db_path)
        except Exception as e:
            logger.warning(f"[EpisodicMemory] 文档存储初始化失败: {e}，将仅使用内存缓存")
            self.doc_store = None

    def add(self, memory_item: MemoryItem) -> str:
        """添加情景记忆"""
        # 从元数据中提取情景信息
        session_id = memory_item.metadata.get("session_id", "default_session")
        context = memory_item.metadata.get("context", {})
        outcome = memory_item.metadata.get("outcome")
        participants = memory_item.metadata.get("participants", [])
        tags = memory_item.metadata.get("tags", [])

        # 创建情景
        episode = Episode(
            episode_id=memory_item.id,
            user_id=memory_item.user_id,
            session_id=session_id,
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=context,
            outcome=outcome,
            importance=memory_item.importance
        )
        self.episodes.append(episode)
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)

        # 持久化存储
        if self.doc_store:
            try:
                ts_int = int(memory_item.timestamp.timestamp())
                self.doc_store.add_memory(
                    memory_id=memory_item.id,
                    user_id=memory_item.user_id,
                    timestamp=ts_int,
                    content=memory_item.content,
                    memory_type="episode",
                    importance=memory_item.importance,
                    properties={
                        "session_id": session_id,
                        "context": context,
                        "outcome": outcome,
                        "participants": participants,
                        "tags": tags
                    }
                )
            except Exception as e:
                logger.warning(f"[EpisodicMemory] 持久化失败: {e}")

        # 向量索引
        if self.vector_store:
            try:
                embedding = self.embedder.encode(memory_item.content)
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                self.vector_store.add_vectors(
                    ids=[memory_item.id],
                    vectors=[embedding],
                    metadata=[{
                        "session_id": session_id,
                        "user_id": memory_item.user_id,
                        "context": str(context),
                        "outcome": outcome or "",
                    }]
                )
            except Exception as e:
                logger.warning(f"[EpisodicMemory] 向量索引失败: {e}")

        return memory_item.id

    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """情景记忆检索"""
        user_id = kwargs.get("user_id")
        session_id = kwargs.get("session_id")
        time_range: Optional[Tuple[datetime, datetime]] = kwargs.get("time_range")
        importance_threshold: Optional[float] = kwargs.get("importance_threshold")

        # 结构化过滤候选集
        candidate_ids: Optional[set] = None
        if (time_range is not None or importance_threshold is not None) and self.doc_store:
            start_ts = int(time_range[0].timestamp()) if time_range else None
            end_ts = int(time_range[1].timestamp()) if time_range else None
            try:
                docs = self.doc_store.search_memories(
                    user_id=user_id,
                    memory_type="episode",
                    start_time=start_ts,
                    end_time=end_ts,
                    importance_threshold=importance_threshold,
                    limit=limit
                )
                candidate_ids = {d["memory_id"] for d in docs}
            except Exception as e:
                logger.warning(f"[EpisodicMemory] 候选集过滤失败: {e}")

        # 向量检索
        hits = []
        if self.vector_store:
            try:
                query_vec = self.embedder.encode(query)
                if hasattr(query_vec, "tolist"):
                    query_vec = query_vec.tolist()
                hits = self.vector_store.search_similar(
                    query_vector=query_vec,
                    top_k=limit * 2,
                )
            except Exception as e:
                logger.warning(f"[EpisodicMemory] 向量检索失败: {e}")

        # 过滤和重排
        now_ts = int(datetime.now().timestamp())
        results: List[Tuple[float, MemoryItem]] = []
        seen = set()

        for hit in hits:
            meta = hit.get("metadata", {})
            mem_id = hit.get("memory_id") or hit.get("id")
            if not mem_id or mem_id in seen:
                continue
            # 检查是否已遗忘
            episode = next((e for e in self.episodes if e.episode_id == mem_id), None)
            if episode and episode.context.get("forgotten", False):
                continue
            if candidate_ids is not None and mem_id not in candidate_ids:
                continue
            if session_id is not None and meta.get("session_id") != session_id:
                continue
            # 从内存缓存或持久化存储读取完整记录
            if episode:
                age_days = max(0.0, (now_ts - int(episode.timestamp.timestamp())) / 86400.0)
                recency_score = 1.0 / (1.0 + age_days)
                vec_score = float(hit.get("score", 0.5))
                base_relevance = vec_score * 0.8 + recency_score * 0.2
                importance_weight = 0.8 + (episode.importance * 0.4)
                final_score = base_relevance * importance_weight
                item = MemoryItem(
                    id=episode.episode_id,
                    content=episode.content,
                    memory_type="episodic",
                    user_id=episode.user_id,
                    timestamp=episode.timestamp,
                    importance=episode.importance,
                    keyword=[],
                    metadata={"session_id": episode.session_id, "relevance_score": final_score}
                )
                results.append((final_score, item))
                seen.add(mem_id)

        # 如果向量检索无结果，回退到内存关键词匹配
        if not results:
            query_lower = query.lower()
            for ep in self._filter_episodes(user_id=user_id, session_id=session_id):
                if query_lower in ep.content.lower():
                    recency_score = 1.0 / (1.0 + max(0.0, (now_ts - int(ep.timestamp.timestamp())) / 86400.0))
                    base_relevance = 0.5 * 0.8 + recency_score * 0.2
                    importance_weight = 0.8 + (ep.importance * 0.4)
                    final_score = base_relevance * importance_weight
                    item = MemoryItem(
                        id=ep.episode_id,
                        content=ep.content,
                        memory_type="episodic",
                        user_id=ep.user_id,
                        timestamp=ep.timestamp,
                        importance=ep.importance,
                        keyword=[],
                        metadata={"session_id": ep.session_id, "relevance_score": final_score}
                    )
                    results.append((final_score, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in results[:limit]]

    def update(self, memory_id: str, content: str = None, importance: float = None,
               metadata: Dict[str, Any] = None) -> bool:
        """更新情景记忆"""
        updated = False
        for episode in self.episodes:
            if episode.episode_id == memory_id:
                if content is not None:
                    episode.content = content
                if importance is not None:
                    episode.importance = importance
                if metadata is not None:
                    episode.context.update(metadata.get("context", {}))
                    if "outcome" in metadata:
                        episode.outcome = metadata["outcome"]
                updated = True
                break

        if self.doc_store:
            try:
                doc_updated = self.doc_store.update_memory(
                    memory_id, content=content, importance=importance, properties=metadata
                )
                updated = updated or doc_updated
            except Exception as e:
                logger.warning(f"[EpisodicMemory] 更新持久化失败: {e}")

        # 重新嵌入
        if content is not None and self.vector_store:
            try:
                embedding = self.embedder.encode(content)
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()
                self.vector_store.add_vectors(
                    vectors=[embedding],
                    ids=[memory_id],
                    metadata=[{"memory_type": "episodic"}]
                )
            except Exception as e:
                logger.warning(f"[EpisodicMemory] 重新索引失败: {e}")

        return updated

    def remove(self, memory_id: str) -> bool:
        """从记忆库中删除情景记忆"""
        removed = False
        for i, episode in enumerate(self.episodes):
            if episode.episode_id == memory_id:
                removed_episode = self.episodes.pop(i)
                sid = removed_episode.session_id
                if sid in self.sessions:
                    try:
                        self.sessions[sid].remove(memory_id)
                    except ValueError:
                        pass
                    if not self.sessions[sid]:
                        del self.sessions[sid]
                removed = True
                break

        if self.doc_store:
            try:
                doc_deleted = self.doc_store.delete_memory(memory_id=memory_id)
                removed = removed or doc_deleted
            except Exception as e:
                logger.warning(f"[EpisodicMemory] 持久化删除失败: {e}")

        if self.vector_store:
            try:
                self.vector_store.delete(ids=[memory_id])
            except Exception:
                pass

        return removed

    # BaseMemory 接口别名
    def delete(self, memory_id: str) -> bool:
        return self.remove(memory_id)

    def has_memory(self, memory_id: str) -> bool:
        """检查记忆库中是否存在指定情景记忆"""
        return any(episode.episode_id == memory_id for episode in self.episodes)

    def clear(self):
        """清空所有情景记忆"""
        self.episodes.clear()
        self.sessions.clear()
        self.patterns_cache.clear()

        if self.vector_store:
            try:
                ids = [ep.episode_id for ep in self.episodes]
                if ids:
                    self.vector_store.delete(ids=ids)
            except Exception:
                pass

    def forget(self, strategy: str = "importance_based", threshold: float = 0.1,
               max_age_days: int = 30) -> int:
        """根据策略遗忘情景记忆(硬删除)"""
        forgotten_count = 0
        current_time = datetime.now()
        to_remove = []

        for episode in self.episodes:
            should_forget = False
            if strategy == "importance_based":
                if episode.importance < threshold:
                    should_forget = True
            elif strategy == "time_based":
                cutoff_time = current_time - timedelta(days=max_age_days)
                if episode.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                if len(self.episodes) > self.config.max_capacity:
                    sorted_episodes = sorted(self.episodes, key=lambda x: x.importance)
                    excess_count = len(self.episodes) - self.config.max_capacity
                    if episode in sorted_episodes[:excess_count]:
                        should_forget = True
            if should_forget:
                to_remove.append(episode.episode_id)

        for episode_id in to_remove:
            if self.remove(episode_id):
                forgotten_count += 1
            logger.info(f"情景记忆硬删除:{episode_id[:8]}...(策略：{strategy})")

        return forgotten_count

    def get_all(self) -> List[MemoryItem]:
        """获取所有情景记忆"""
        return [
            MemoryItem(
                id=ep.episode_id,
                content=ep.content,
                memory_type="episodic",
                importance=ep.importance,
                user_id=ep.user_id,
                timestamp=ep.timestamp,
                keyword=[],
                metadata={"session_id": ep.session_id, "context": ep.context, "outcome": ep.outcome}
            )
            for ep in self.episodes
        ]

    def get_stats(self) -> Dict[str, Any]:
        """获取情景记忆统计信息"""
        active_episodes = self.episodes
        vs_stats = {}
        if self.vector_store:
            try:
                vs_stats = self.vector_store.get_stats()
            except Exception:
                vs_stats = {"store_type": "Milvus"}

        return {
            "count": len(active_episodes),
            "forgotten_count": 0,
            "total_count": len(self.episodes),
            "sessions_count": len(self.sessions),
            "avg_importance": (
                sum(e.importance for e in active_episodes) / len(active_episodes)
                if active_episodes else 0.0
            ),
            "time_span_days": self._calculate_time_span(),
            "memory_type": "episodic",
            "vector_store": vs_stats,
        }

    def find_patterns(self, user_id: str = None, min_frequency: int = 2) -> List[Dict[str, Any]]:
        """发现用户行为模式"""
        cache_key = f"{user_id}_{min_frequency}"
        if (cache_key in self.patterns_cache and self.last_pattern_analysis and
                (datetime.now() - self.last_pattern_analysis).seconds < 3600):
            return self.patterns_cache[cache_key]

        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        keyword_patterns: Dict[str, int] = {}
        context_patterns: Dict[str, int] = {}

        for episode in episodes:
            words = episode.content.lower().split()
            for word in words:
                if len(word) > 3:
                    keyword_patterns[word] = keyword_patterns.get(word, 0) + 1
            for key, value in episode.context.items():
                pk = f"{key}:{value}"
                context_patterns[pk] = context_patterns.get(pk, 0) + 1

        patterns = []
        for keyword, frequency in keyword_patterns.items():
            if frequency >= min_frequency:
                patterns.append({
                    "type": "keyword",
                    "pattern": keyword,
                    "frequency": frequency,
                    "confidence": frequency / len(episodes) if episodes else 0
                })
        for cp, frequency in context_patterns.items():
            if frequency >= min_frequency:
                patterns.append({
                    "type": "context",
                    "pattern": cp,
                    "frequency": frequency,
                    "confidence": frequency / len(episodes) if episodes else 0
                })

        patterns.sort(key=lambda x: x["frequency"], reverse=True)
        self.patterns_cache[cache_key] = patterns
        self.last_pattern_analysis = datetime.now()
        return patterns

    def get_timeline(self, user_id: str = None, Limit: int = 50) -> List[Dict[str, Any]]:
        """获取时间线视图"""
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        episodes.sort(key=lambda x: x.timestamp, reverse=True)
        return [
            {
                "episode_id": ep.episode_id,
                "timestamp": ep.timestamp.isoformat(),
                "content": ep.content,
                "session_id": ep.session_id,
                "importance": ep.importance,
                "outcome": ep.outcome
            }
            for ep in episodes[:Limit]
        ]

    def _filter_episodes(self, user_id: str = None, session_id: str = None,
                         time_range: Tuple[datetime, datetime] = None) -> List[Episode]:
        """过滤情景记忆"""
        filtered = self.episodes
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        if session_id:
            filtered = [e for e in filtered if e.session_id == session_id]
        if time_range:
            start_time, end_time = time_range
            filtered = [e for e in filtered if start_time <= e.timestamp <= end_time]
        return filtered

    def _calculate_time_span(self) -> float:
        """计算时间跨度（天数）"""
        if not self.episodes:
            return 0.0
        return (max(e.timestamp for e in self.episodes) - min(e.timestamp for e in self.episodes)).days
