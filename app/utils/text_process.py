from abc import abstractmethod
from datetime import datetime
import json
import re
import uuid
from typing import List
from langdetect import detect, DetectorFactory, LangDetectException
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy.orm.strategies import SelectInLoader
from sympy import limit

from app.agent.agent import Agent
from app.llm.select_llm import LLM
from app.memory.baseMemory import MemoryItem
from app.prompt.default_prompt import INTENT_PROMPT

text_process_llm = LLM(model="gemini-2.5-flash",provider="modelscope")

class SessionUserId:
    def __init__(self,user_id:str=None):
        self.user_id = user_id if user_id else str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())

class TextProcess:
    def __init__(self,
                 llm:Agent
                 ):
        self.llm = llm

    def content_to_memory(self,
                          text: str,
                          user_id: str,
                          ) -> MemoryItem:
        return MemoryItem(
            id=str(uuid.uuid4()),
            content=self.abstract_user_intent(text),
            keyword=self.extract_keywords(text),
            memory_type="workingmemory",
            user_id=SessionUserId().user_id,
            timestamp=datetime.now(),
            importance=0.5  # 使用默认重要性

        )
    def abstract_user_intent(self, text:str):
        prompt = INTENT_PROMPT.format(user_input=text)
        llm_input = [{"role":"user","content":prompt}]
        intent_json = self.llm.invoke(llm_input)
        try:
            data = json.loads(intent_json)
            data = data["intent"]
        except json.JSONDecodeError:
            #使用规则+模版
            templates = [
                ("想做","用户希望构建"),
                ("我要做","用户希望构建"),
                ("我想","用户希望"),
                ("帮我","用户希望构建")
            ]
            abstract = text
            for k,v in templates:
                abstract = abstract.replace(k,v)
            data  = {
                "intent":abstract,
                "contraints":[],
            }
        return data

    def extract_keywords(self, text,top_k=5)->List[str]:

        # lang = self.detect_language(text)
        # if lang == "zh":
        #     tokenizer = self.zh_tokenize
        # else:
        #     tokenizer = self.en_tokenize
        #
        #
        # vectorizer = TfidfVectorizer(
        #     tokenizer = tokenizer,
        #     token_pattern = None,
        #     # max_df = 0.8,
        #     # min_df = 0.4,
        #     ngram_range=(1,2)
        # )
        # tfidf = vectorizer.fit_transform([text])
        # scores = tfidf.toarray()[0]
        # terms = vectorizer.get_feature_names_out()
        #
        # ranked = sorted(
        #     zip(terms, scores),
        #     key = lambda x:x[1],
        #     reverse = True
        # )
        # return [term for term,_ in ranked[:top_k]]
        keyword_prompt="""
        你现在的任务是负责提取下面用户输入的关键词,关键词以在语义中的重要性排序输出。
        输出格式:
        'keyword_1','keyword_2','keyword_3','keyword_4','keyword_5'...
        用户输入:
        {user_input}
        
        """
        message = [{"role":"user","content":keyword_prompt.format(user_input=text)}]
        response = text_process_llm.invoke(message)
        keywords =response.split(',')
        return keywords[:top_k]

    def zh_tokenize(self,text:str):
        return [w for w in jieba.lcut(text) if len(w.strip()) > 1]

    def en_tokenize(self,text:str):
        return re.findall(f"[A-Za-z][A-Za-z0-9_-]+]",text)

    def detect_language(self,text:str)->str:
        """ 返回:'zh'|'en' """
        try:
            lang = detect(text)
        except LangDetectException:
            #极短文本/噪声文本
            return "zh" #中文系统里默认中文更安全
        #langdetect 返回的可能是：zh-cn/zh-tw/en/ja/fr
        if lang.startswith("zh"):
            return "zh"
        else:
            return "en"
