"""
情景记忆实现
提供:
- 具体交互事件存储
- 时间序列组织
- 上下文丰富的记忆
- 模式识别能力

"""
from app.utils.milvus_store import MilvusConnectionManager
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from app.memory.BaseMemory import MemoryConfig,BaseMemory,MemoryItem
from app.utils.document_store import PostgreSQLDocumentStore
from app.model.embedder import get_text_embedder

class Episode:
    """情景记忆中的单个场景
    """
    def __init__(self,
                 episode_id:str,
                 user_id:str,
                 session_id:str,
                 timestamp:datetime,
                 content:str,
                 context:Dict[str,Any],
                 outcome:Optional[str]=None,
                 importance:float=0.5
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
    """
    def __init__(self,config:MemoryConfig,storage_backend = None):
        super.__init__(config,storage_backend)

        #本地缓存(内存)
        self.episodes:List[Episode] = []
        self.sessions:Dict[str,List[str]]={}#session_id->spisode_ids

        #模式识别缓存
        self.patterns_cache = {}
        self.last_pattern_analysis = None

        #权威文档存储(PostgreSQL)
        db_dir = self.config.storage_path if hasattr(self.config,"storage_path") else ".memory_data"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir,"memory.db")
        self.doc_store = PostgreSQLDocumentStore(db_path=db_path)

        #统一嵌入模型(多语言，默认384维)
        self.embedder = get_text_embedder()


        #向量存储(Milvus)
        from pymilvus import MilvusClient
        milvus_url = os.getenv("MILVUS_URL")
        milvus_api_key = os.getenv("MILVUS_API_KEY")
        self.vector_store = MilvusConnectionManager.get_instance(
            url=milvus_url,
            collection_name="episodes",
            metric_type="COSINE",
            vector_size=384,
        )

    def add(self,memory_item:MemoryItem)->str:
        """添加情景记忆"""
        #从元数据中提取情景信息
        session_id = memory_item.metadata.get("session_id","default_session")
        context = memory_item.metadata.get("context",{})
        outcome = memory_item.metadate.get("outcome")
        participants = memory_item.metadata.get("participants",[])
        tags = memory_item.metadata.get("tags",[])

        #创建情景
        episode = Episode(
            episode_id = memory_item.id,
            user_id = memory_item.user_id,
            session_id = session_id,
            timestamp= = memory_item.timestamp,
            content = memory_item.content,
            context = context,
            outcome = outcome,
            importance = memory_item.importance
        )
        self.episodes.append(episode)
        if session_id not in self.sessions:
            self.sessions[session_id]=[]

        self.sessions[session_id].append(episode.episode_id)

        #使用权威存储
        ts_int = int(memory_item.timestamp.timestamp())
        self.doc_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            timestamp=ts_int,
            content = memory_item.content,
            memory_type="episode",
            importance = memory_item.importance,
            properties = {
                "session_id":session_id,
                "context":context,
                "outcome":outcome,
                "participants":participants,
                "tags":tags
            }

        )
        #向量索引
        try:
            embedding = self.embedder.encode(memory_item.content)
            if hasattr(embedding,"tolist"):
                embedding = embedding.tolist()
            self.vector_store.add_vectors(
                ids=[memory_item.id],
                vectors=[embedding],
                metadata=[{
                    "session_id":session_id,
                    "context":context,
                    "outcome":outcome,
                    "participants":participants,
                    "tags":tags
                }]                        
            )
        except Exception:
            #向量入库失败不影响权威存储
            pass 
        
        return memory_item.id

    def retrieve(self,query:str,limit:int=5,**kwargs)->List[MemoryItem]:

        """情景记忆检索"""
        user_id = kwargs.get("user_id")
        session_id = kwargs.get("session_id")
        time_range:Optional[Tuple[datetime,datetime]] = kwargs.get("time_range")
        importance_threshold:Optional[Tuple[datetime,datetime]]=kwargs.get("importance_threshold")


        #结构化过滤后选
        candidate_ids:Optional[set]=None
        if time_range is not None or importance_threshold is not None:
            start_ts = int(time_range[0].timestamp()) if time_range else None
            end_ts = int(time_range[1].timestamp()) if time_range else None
            docs = self.doc_store.search_memories(
                user_id=user_id,
                memory_type="episode",
                start_time=start_ts,
                end_time = end_ts,
                importance_threshold=importance_threshold,
                limit=limit
                )
            candidate_ids = {d["memory_id"] for d in docs}

        #向量检索
        try:
            query_vec = self.embedder.encode(query)
            if hasattr(query_vec,"tolist"):
                query_vec = query_vec.tolist()
            filter_exp = f"memory_type == 'episode' AND user_id == {user_id}"

            hits = self.vector_store.search_similar(
                query_vector=query_vec,
                top_k=limit,
                filter_expr=filter_exp
            )
        except Exception:
            logger.info("Milvus检索失败")
            hits = []

        #过滤和重排
        now_ts = int(datetime.now().timestamp())
        results:List[Tuple[float,MemoryItem]]=[]
        seen = set()
        for hit in hits:
            meta = hit.get("metadata",{})
            mem_id = hit.get("memory_id")
            if not mem_id or mem_id in seen:
                continue
            #检查是否遗忘
            episode = next((e for e in self.episodes if e.episode_id==mem_id),None)
            if episode and episode.context.get("forgotten",False):
                continue#跳过已经遗忘的记忆

            if candidate_ids is not None and mem_id not in candidate_ids:
                continue#如果候选记忆不是空和记忆id不在候选集中，跳过
            if session_id is not None and meta.get("session_id")!=session_id:
                continue#如果会话id不为空且记忆的会话id与查询的会话id不匹配，跳过

            #从权威库读取完整记录
            doc = self.doc_store.get_memory(mem_id)
            if doc is None:
                continue
            #计算综合分数：向量0.6+近因0.2+重要性0.2
            vec_score = float(hit.get("score",0.0))
            age_days = max(0.0,(now_ts-int(doc["timestamp"]))/86400.0)
            recency_score = 1.0/(1.0+age_days)
            importance_score = float(doc.get("importance",0.5))

            #新评分算法：向量检索纯基于相似度，重要性作为加权因子
            #基础相似度得分(不受重要性影响)
            base_relevance = vec_score * 0.8 + recency_score * 0.2

            #重要性作为加权因子，范围[0.8,1.2]
            importance_weight = 0.8 + (importance_score * 0.4)

            #最终得分: 相似度*重要性权重
            final_score = base_relevance * importance_weight
            item = MemoryItem(
                id = doc["memory_id"],
                user_id = doc["user_id"],
                timestamp = datetime.fromtimestamp(doc["timestamp"]),
                content = doc["content"],
                importance = doc.get("importance",0.5),
                memory_type=doc.get("memory_type"),
                metadate = {
                    **doc.get("properties",{}),
                    "relevance_score":final_score,
                    "vector_score":vec_score,
                    "recency_score":recency_score
                }
            )
            results.append((final_score,item))
            seen.add(mem_id)
        
        #如果向量检索无结果，回退到简单关键词匹配(内存缓存)
        if not results:
            fallback = super()._generate_id#占位以避免未使用警告
            query_lower = query.lower()
            for ep in self._filter_episodes(user_id,session_id,time_range):
                if query_lower in ep.context.lower():
                    recency_score = 1.0/(1.0+max(0.0,(now_ts-int(ep.timestamp.timestamp()))/86400.0))
                    #回退匹配
                    keyword_score = 0.5 #简单关键词匹配的基础分数
                    base_relevance = keyword_score*0.8+recency_score*0.2
                    importance_weight = 0.8+(ep.importance*0.4)

                    final_score = base_relevance*importance_score
                    item = MemoryItem(

                        id = ep.episode_id,
                        content = ep.content,
                        memory_type="episodic",
                        user_id = ep.user_id,
                        timestamp=ep.timestamp,
                        importance = ep.importance,
                        metadate={
                            "session_id":ep.session_id,
                            "context":eq.context,
                            "outcome":eq.outcome,
                            "relevance_score:":final_score
                        }
                    )
                    results.append((final_score,item))
        results.sort(key=lambda x:x[0],reverse=True)
        return [it for _,it in results[:limit]]

    def update(self,memory_id:str,content:str=None,importane:float = None,metadata:Dict[str,Any]=None)->bool:
        """更新情景记忆(PostgreSQL为权威，Milvus按需重嵌入)"""
        
        updated = False
        for episode in self.episodes:
            if episode.episode_id == memory_id:
                if content is not None:
                    episode.content = content
                if importance is not None:
                    episode.importance = importance
                if metadata is not None:
                    episode.context.update(metadata.get("context",{}))
                    if "outcome" in metadata:
                        episode.outcome = metadata["outcome"]
                updated = True
                break
        #更新到PostgreSQL
        doc_updated = self.doc_store.update_memory(memory_id,content = content,importance = importance,properties = metadata)

        #如果内容发生改变,重新嵌入Milvus
        if content is not None:
            try:
                embedding = self.embedder.encode(content)
                if hasattr(embedding,"tolist"):
                    embedding = embedding.tolist()
                #获取更新后的记录以同步payload
                doc = self.doc_store.get_memory(memory_id)
                payload = {
                    "memory_id":memory_id,
                    "user_id":doc["user_id"] if doc else "",
                    "memory_type":"episodic",
                    "importance":(doc.get("importance")if doc else importane) or 0.5,
                    "session_id":(doc.get("properties",{}) or {}).get("session_id"),
                    "content":content
                }
                self.vector_store.add_vectors(
                    vectors=[embedding],
                    ids=[memory_id],
                    metadatas=[payload]
                )
            except Exception:
                pass
        return updated or doc_updated

    
    def remove(self,memory_id:str)->bool:
        """从记忆库中删除情景记忆"""
        removed = False
        for i,episode in enumerate(self.episodes):
            if episode.episode_id == memory_id:
                removed_episode = self.episodes.pop(i)
                session_id = removed_episode.session_id
                if session_id in self.sessions:
                    self.sessions[session_id].remove(memory_id)
                    if not self.sessions[session_id]:
                        del self.sessions[session_id]
                removed = True
                break
        # 权威库删除
        doc_deleted = self.doc_store.delete_memory(memory_id=memory_id)

        #向量库删除
        try:
            self.vector_store.delete(ids=[memory_id])
        except Exception:
            pass
        return removed or doc_deleted

    def has_memory(self,memory_id:str)->bool:
        """检查记忆库中是否存在指定情景记忆"""
        return any(episode.episode_id == memory_id for episode in self.episodes)
    

    def clear(self):
        """清空所有情景记忆(仅清理episodic，不影响其他类型)"""
        #内存缓存
        self.episodes.clear()
        self.sessions.clear()
        self.patterns_cache.clear() 

        #SQLite内的episodic全部删除
        docs = self.doc_store.search_memories(memory_type="episodic",limit=10000) 
        ids = [d["memory_id"] for d in docs]
        for mid in ids:
            self.doc_store.delete_memory(memory_id=mid)

        #Milvus内的episodic全部删除
        try:
            if ids:
                self.vector_store.delete(ids=ids)
        except Exception:
            pass

    def forget(self,strategy:str = "importance_based",threshold:float=0.1,max_age_days:int=30)->int:
        """根据策略遗忘情景记忆(硬删除)"""
        forgotten_count = 0
        current_time = datetime.now()

        to_remove = []#收集要删除的记忆ID
        for episode in self.episodes:
            should_forget = False
            if strategy == "importance_based":
                #基于重要性遗忘
                if episode.importance<threshold:
                    should_forget = True
            elif strategy == "time_based":
                #基于时间遗忘
                cutoff_time = current_time - timedelta(days = max_age_days)
                if episode.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                #基于容量遗忘
                if len(self.episodes)>self.config.max_capacity:
                    if len(self.episodes)>self.config.max_capacity:
                        sorted_episodes = sorted(self.episodes,key=lambda x:x.importance)
                        excess_count = len(self.episodes)-self.config.max_capacity
                        if episode in sorted_episodes[:excess_count]:
                            should_forget = True

            if should_forget:
                to_remove.append(episode.episode_id)
                forgotten_count += 1

        #执行硬删除
        for episode_id in to_remove:
           if self.remove(episode_id):
                forgotten_count += 1
            logger.info(f"情景记忆硬删除:{episode_id[:8]}...(策略：{strategy})")
        return forgotten_count

    def get_all(self)->List[MemoryItem]:
        """获取所有情景记忆"""
        memory_items=[]
        for episode in self.episodes:
            memory_item = MemoryItem(
                id = episode.episode_id,
                content = episode.content,
                memory_type = "episodic",
                importance = episode.importance,
                user_id= episode.user_id,
                timestamp=episode.timestamp,
                importance = episode.importance,
                metadata=episode.metadata
            )
            memory_items.append(memory_item)
        return memory_items
    def get_stats(self)->Dict[str,Any]:
        """获取情景记忆统计信息(合并PostgreSQL与Milvus)"""
        #硬删除模式：所有episodes都是活跃的
        active_episodes = self.episodes

        db_stats = self.doc_store.get_database_stats()
        try:
            vs_stats = self.vector_store.get_stats()
        except Exception:
            vs_stats ={"store_type":"Milvus"}
        return {

            "count":len(active_episodes),#活跃记忆数量
            "forgotten_count":0,#硬删除模式下已遗忘的记忆会被直接删除
            "total_count":len(self.episodes),#总记忆数量
            "sessions_count":len(self.sessions),#会话数量
            "avg_importance":sum(e.importance for e in active_episodes)/len(active_episodes) if active_episodes else 0.0,
            "time_span_days":self._calculate_time_span(),
            "memory_type":"episodic",
            "vector_store":vs_stats,
            "document_store":{k ,v for k,v in db_stats.items() if k.endswith("_count") or k in ["store_type","db_path"]}
        }

    def find_pattterns(self,user_id:str=None,min_frequency:int=2)->List[Dict[str,Any]]:
        """发现用户行为模式"""
        #检查缓存
        cache_key = f"{user_id}_{min_frequency}"
        if (cache_key in self.patterns_cache and self.last_pattern_analysis and (datetime.now()-self.last_pattern_analysis).hours<1):
            return self.patterns_cache[cache_key]
        
        
        #过滤情景
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]

        #简单的模式识别：基于内容关键词
        keyword_patterns = {}
        context_patterns = {}

        for episode in episodes:
            #提取关键词
            words = episode.content.lower().split()
            for word in words:
                if len(word)>3:#忽略短词
                    keyword_patterns[word] = keyword_patterns.get(word,0)+1
            
            #提取上下文模式
            for key,value in episode.context.items():
                pattern_key = f"{key}:{value}"
                context_patterns[pattern_key] = context_patterns.get(pattern_key,0)+1
        
        #筛选频繁模式
        patterns = []
        
        for keyword,frequency in keyword_patterns.items():
            if frequency>=min_frequency:
                patterns.append({
                    "type":"keyword",
                    "pattern"keyword,
                    "frequency":frequency,
                    "confidence":frequency/len(episodes)
                })
        for context_pattern,frequency in context_patterns.items():
            if frequency>=min_frequency:
                patterns.append({
                    "type":"context",
                    "pattern":context_pattern,
                    "frequency":frequency,
                    "confidence":frequency/len(episodes)
                })


        #按频率排序
        patterns.sort(key=lambda x:x["frequency"],reverse=True)

        #缓存结果
        self.patterns_cache[cache_key] = patterns
        self.last_pattern_analysis = datetime.now()
        return patterns


    def get_timeline(self,user_id:str=None,Limit:int=50)->List[Dict[str,Any]]:
        """获取时间线视图"""
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]

        episodes.sort(key=lambda x:x.timestamp,reverse=True)
        
        timeline = []
        for episode in episodes[:limit]:
            timeline.append({
                "episode_id":episode.episode_id,
                "timestamp":episode.timestamp,
                "content":episode.content,
                "session_id":episode.session_id,
                "importance":episode.importance,
                "outcome":episode.outcome
            })

        return timeline
    
    def _filter_episodes(
        self,
        user_id:str=None,
        session_id:str=None,
        time_range:Tuple[datetime,datetime]=None
    )->List[Episode]:
        """过滤情景记忆"""
        filtered = self.episodes
        if user_id:
            filtered = [e for e in filtered if e.user_id==user_id]
        
        if session_id:
            filtered = [e for e in filtered if e.session_id==session_id]
        if time_range:
            start_time,end_time = time_range
            filtered = [e for e in filtered if start_time<=e.timestamp<=end_time]

        return filtered
    
    def _calculate_time_span(self)->float:
        """计算时间跨度"""
        if not self.episodes:
            return 0.0
        return (max(e.timestamp for e in self.episodes)-min(e.timestamp for e in self.episodes)).days


    def _persist_episode(self,episode:Episode)->bool:
        """持久化情景记忆"""
        if self.storage and hasattr(self.storage,"add_memory"):
            self.storage.add_memory(
                memory_id = episode.episode_id,
                user_id = episode.user_id,
                content = episode.content,
                memory_type = "episodic",
                timestamp = int(episode.timestamp.timestamp()),
                importance = episode.importance,
                properties = {
                    "session_id":episode.session_id,
                    "outcome":episode.outcome,
                    "context":episode.context
                }
            )

    def _remove_from_storage(self,memory_id:str):
        """从存储中删除记忆"""
        if self.storage and hasattr(self.storage,"delete_memory"):
            self.storage.delete_memory(memory_id)








        



