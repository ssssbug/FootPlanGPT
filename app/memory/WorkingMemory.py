import heapq
from datetime import datetime, timedelta
from typing import List, Dict, Any

from qdrant_client.http.models import DecayParamsExpression

from app.memory.baseMemory import BaseMemory, MemoryConfig, MemoryItem


class WorkingMemory(BaseMemory):


    """工作记忆实现"
    特点：
    - 容量有限(默认50条)+TTL自动清理
    - 纯内存存储，访问速度快
    - 混合检索:TF-IDF向量化+关键词匹配
    - 优先级管理
    """
    def __init__(self,config:MemoryConfig,storage_backend=None):
        super().__init__(config,storage_backend)
        self.max_capacity = config.max_capacity
        self.max_tokens = config.max_tokens
        #纯内存TTL
        self.max_save_minutes = getattr(config,'workingmemory_ttl_minutes',120)
        self.current_tokens = 0
        self.session_start = datetime.now()

        #内存存储
        self.memories:List[MemoryItem]=[]

        #使用优先级队列管理记忆
        self.memory_heap = []#(priority,timestamp,memory_item)

    def add(self,item:MemoryItem):
        """添加工作记忆"""
        self.expire_old_memories()#清除过期记忆
        #计算优先级
        priority = self.calculate_priority(item)
        heapq.heappush(self.memory_heap,(-priority,item.timestamp,item))
        self.memories.append(item)

        #更新token数
        self.current_tokens+=len(item.content.split())
        #检查容量限制
        self.enforce_capacity_limits()

        return item.id

    def retrieve(self,query:str,limit:int=5,user_id:str=None,**kwargs) ->List[MemoryItem]:
        """混合检索：混合语义向量检索+关键词匹配"""
        self.expire_old_memories()
        if not self.memories:
            return []
        #过滤已经遗忘的记忆
        active_memories = [m for m in self.memories if not m.metadata.get("forgotten",False)]
        #按用户ID过滤
        filtered_memories = active_memories
        if user_id:
            filtered_memories = [m for m in active_memories if m.user_id == user_id]
        if not filtered_memories:
            return []

        #使用语义向量检索
        vector_scores = {}
        try:
            #简单的语义相似度计算
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            #准备内容
            documents = [query]+[m.content for m in filtered_memories]

            #TF-IDF向量化
            vectorizer = TfidfVectorizer(stop_words=None,lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(documents)

            #计算相似度
            query_vector = tfidf_matrix[0:1]
            doc_vector = tfidf_matrix[1:]
            similarity = cosine_similarity(query_vector,doc_vector).flatten()

            #存储向量分数
            for i,memory in enumerate(filtered_memories):
                vector_scores[memory.id] = similarity[i]
        except Exception as e:
            #若果向量检索失败,回退到关键词匹配
            vector_scores = {}

        #计算最终分数
        query_lower = query.lower()
        scored_memories = []
        for mem in filtered_memories:
            content_lower = mem.content.lower()
            #获取向量分数
            vector_score = vector_scores.get(mem.id,0.0)
            #关键词匹配分数
            keyword_score = 0.0
            if query_lower in content_lower:
                keyword_score = len(query_lower)/len(content_lower)
            else:
                #分词匹配
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                intersection = query_words.intersection(content_words)
                if intersection:
                    keyword_score = len(intersection)/len(query_words.union(content_words))*0.8

            #混合分数：向量检索+关键词匹配
            if vector_score>0:
                base_relevance = vector_score*0.7+keyword_score*0.3
            else:
                base_relevance = keyword_score
            #时间衰减
            time_decay=self.calculate_time_decay(mem.timestamp)
            base_relevance *= time_decay

            #重要性权重
            importance_weight = 0.8+(mem.importance*0.8)
            final_score = base_relevance*importance_weight
            if final_score>0:
                scored_memories.append((final_score,mem))
        #按分数排序并返回
        scored_memories.sort(key=lambda x:x[0],reverse=True)
        return [mem for _,mem in scored_memories[:limit]]

    def update(self,memory_id:str,content:str=None,importance:float=None,metadata:Dict[str,Any]=None) ->bool:
        """更新工作记忆"""
        for mem in self.memories:
            if mem.id==memory_id:
                old_tokens = len(mem.content.split())
                if content is not None:
                    mem.content = content
                    #更新token数
                    new_tokens = len(content.split())
                    self.current_tokens=self.current_tokens+new_tokens-old_tokens
                if importance is not None:
                    mem.importance=importance
                if metadata is not None:
                    mem.metadata.update(metadata)
                #重新计算优先级并更新堆
                self.update_heap_priority(mem)
                return True
        return False
    def remove(self, id):
        """根据记忆id删除记忆"""
        for i,mem in enumerate(self.memories):
            if mem.id==id:
                #从列表中删除
                removed_memory = self.memories.pop(i)

                #从堆中标记删除
                self.mark_deleted_in_heap(id)
                #更新token数
                self.current_tokens -= len(mem.content.split())
                self.current_tokens =max(0,self.current_tokens)

                return True
        return False
    def has_memory(self,memory_id:str) ->bool:
        if not self.memories:
            return False
        for i,mem in enumerate(self.memories):
            if mem.id==memory_id:
                return True

    def delete(self,memory_id:str) ->bool:
        if not self.memories:
            return False
        self.memories = [m for m in self.memories if m.id!=memory_id]
        return True
    def clear(self):
        """清空所有记忆"""
        self.memories.clear()
        self.memory_heap.clear()
        self.current_tokens = 0
        print("所有记忆已经清空")

    def get_stats(self) ->Dict[str, Any]:
        """获取工作记忆所有统计信息"""
        #过期清理
        self.expire_old_memories()
        active_memories = self.memories
        return {
            "count":len(self.memories),
            "forgotten_count":0,
            "total_count":len(self.memories),
            "max_capacity": self.max_capacity,
            "max_age_minutes": self.max_age_minutes,
            "session_duration_minutes": (datetime.datetime.now() - self.session_start).total_seconds() / 60,
            "avg_importance": sum(m.importance for m in active_memories) / len(
                active_memories) if active_memories else 0.0,
            "capacity_usage": len(active_memories) / self.max_capacity if self.max_capacity > 0 else 0.0,
            "token_usage": self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0,
            "memory_type": "working"
        }
    def get_recent(self,limit:int=10)->List[MemoryItem]:
        """获取最近的记忆"""
        sorted_memories = sorted(self.memories,key=lambda x:x.timestamp,reverse=True)
        return sorted_memories[:limit]

    def get_importance(self,limit:int=10)->List[MemoryItem]:
        """获取重要记忆"""
        sorted_memories = sorted(self.memories,key=lambda x:x.importance,reverse=True)
        return sorted_memories[:limit]
    def get_all(self)->List[MemoryItem]:
        return self.memories.copy()

    def get_context_summary(self,max_length:int=1000)->str:
        """获取上下文摘要
        使用智能压缩策略（memoru中包含关键词的关键句）
        还可以优化，目前是选前10条最重要的，会有压缩后不到max_length的情况
        """
        if not self.memories:
            return "No working memories available"
        #按时间和重要性排序
        sorted_memories = sorted(self.memories,key=lambda x:(x.importance,x.timestamp),reverse=True)

        ##压缩memory
        limit_memory = sorted_memories[:10]#获取前10个记忆
        #压缩每个记忆的内容
        compress_parts=[]
        total_length = 0
        for mem in limit_memory:
            if total_length>=max_length:
                break
            compress_mem = self.compress_memory(mem,max_length-total_length)
            if compress_mem and len(compress_mem)+total_length<=max_length:
                compress_parts.append(compress_mem)
                total_length+=len(compress_mem)
        #生成摘要
        if not compress_parts:
            return "No memories could be summarized with length limit"
        summary = "Working Memories Context:\n".join(compress_parts)
        #如果还有空间
        if len(summary)<max_length-100:
            #添加统计信息
            stats=f"\n\n[Total:{len(self.memories)} memories,{self.current_tokens} tokens, Top {len(compress_parts)} shown]"
            if len(summary)+len(stats)<=max_length:
                summary+=stats
        return summary









    def expire_old_memories(self) :
        """按照TTL+重要性清理过期记忆,并同步更新堆与token数"""
        if not self.memories:
            return

        cutoff = datetime.now()-timedelta(minutes=self.max_save_minutes)
        removed_token_sum=0
        #首先 在这个cutoff时间里找出重要性最小的
        min_impotance=1.0
        global_min_importance = 1.0
        for m in self.memories:
            if m.timestamp<cutoff:
                if m.impotance<=min_impotance:
                    min_impotance=m.impotance
            else:
                if m.impotance<=global_min_importance:
                    global_min_importance=m.impotance

        #保存重要性在min_importance和global_min_importance两侧的记忆
        keep_memory = []
        for m in self.memories:
            if global_min_importance<=min_impotance:
                if m.importance<=global_min_importance or m.impotance>=min_impotance:
                    keep_memory.append(m)
                else:
                    removed_token_sum+=len(m.content.split())

            else:
                if m.impotance>=global_min_importance or m.impotance<=min_impotance:
                    keep_memory.append(m)
                else:
                    removed_token_sum+=len(m.content.split())
        #覆盖列表与token
        self.memories=keep_memory
        self.current_tokens=max(0,self.current_tokens-removed_token_sum)

        #重建堆
        self.memory_heap=[]
        for mem in self.memories:
            priority = self.calculate_priority(mem)
            heapq.heappush(self.memory_heap,(-priority,mem.timestamp,mem))


    def calculate_priority(self,mem:MemoryItem)->float:
        """计算记忆优先级"""
        #基础优先级=重要性
        priority = mem.importance

        #时间衰减
        time_decay = self.calculate_time_decay(mem.timestamp)
        priority *= time_decay
        return priority

    def calculate_time_decay(self,timestamp:datetime)->float:
        """计算时间衰减因子"""
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds()/3600

        #指数衰减
        decay_factor = self.config.decay_factor**(hours_passed/6)
        return max(0.1,decay_factor)

    def enforce_capacity_limits(self):
        """强制执行容量限制"""
        while len(self.memories)>self.max_capacity:
            self._remove_lowest_priority_memory()
        #检查token限制
        while self.current_tokens>self.max_tokens:
            self._remove_lowest_priority_memory()

    def _remove_lowest_priority_memory(self):
        """删除最低优先级的记忆"""
        if not self.memories:
            return

        lowest_priority = float('inf')
        lowest_memory = None
        for mem in self.memories:
            if mem.priority<lowest_priority:
                lowest_priority = mem.priority
                lowest_memory = mem
        if lowest_memory:
            self.remove(lowest_memory.id)



    def mark_deleted_in_heap(self, id):
        pass

    def update_heap_priority(self, mem):
        """更新堆中记忆的优先级"""
        self.memory_heap=[]
        for mem in self.memories:
            priority = self.calculate_priority(mem)
            heapq.heappush(self.memory_heap,(-priority,mem.timestamp,mem))

    def compress_memory(self, mem:MemoryItem, limit:int)->str:
        """压缩单条记忆"""
        if len(mem.content)<limit:
            return mem.content

        #寻找memories中包含关键词的关键局
        if hasattr(mem,"keyword") and mem.keyword:
            keyword_str= ",".join(mem.keyword)

            #获取包含关键词的关键局
            key_sentence = self.extract_key_sentence(mem.content,keyword_str)
            if key_sentence and len(key_sentence)<=limit:
                #如果关键句失败并且关键句子小于限制长度
                return f"[{keyword_str}]{key_sentence}"
            elif len(keyword_str)<=limit:
                #关键词长度小于限制长度
                return f"[Keywords:{keyword_str}]"
        #智能截断（在句子边界截断）
        return self.smart_truncate(mem.content,limit)

    def extract_key_sentence(self, content:str, keyword_str:str)->str:
        """提取包含关键词的关键句子"""
        sentences = content.split(".")

        for sentence in sentences:
            if any(keyword in sentence for keyword in keyword_str.split(",")):
                cleaned = sentence.strip()
                if 20<=len(cleaned)<=150:#合理的句子长度
                    return cleaned+"."


        #返回第一个完整的句子
        for sentence in sentences:
            cleaned = sentence.strip()
            if len(cleaned)>=10:
                return cleaned[:100]+("..." if len(cleaned)>=100 else "")
        return content[:100]+"..." if len(content)>=100 else content

    def smart_truncate(self, content, limit):
        """智能截断：在句子或单词边界截断"""
        if len(content)<=limit:
            return content

        #尝试在句子边界截断
        if '.' in content[:limit]:
            last_dot = content[:limit].rfind('.')
            if last_dot>limit*0.5:#确保保留足够内容
                return content[:last_dot]+"..."

        #强制截断
        return content[:limit-3]+"..."

















































