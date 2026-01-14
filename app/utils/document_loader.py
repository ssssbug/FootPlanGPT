
import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

# 尝试导入 pypdf，如果不存在则提示
try:
    import pypdf
except ImportError:
    pypdf = None

@dataclass
class Document:
    """文档原始数据类"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        if self.doc_id is None:
            # 基于内容生成确定性ID
            self.doc_id = hashlib.md5(self.content.encode('utf-8')).hexdigest()

@dataclass
class DocumentChunk:
    """文档切片类"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_index: int = 0

    def __post_init__(self):
        if self.chunk_id is None:
            # 基于文档ID、索引和部分内容生成ID
            raw = f"{self.doc_id}_{self.chunk_index}_{self.content[:50]}"
            self.chunk_id = hashlib.md5(raw.encode('utf-8')).hexdigest()

class DocumentProcessor:
    """文档处理与分块器"""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 默认分隔符，优先级从高到低
        self.separators = separators or ["\n\n", "\n", "。", ".", "，", ",", " "]

    def process_document(self, document: Document) -> List[DocumentChunk]:
        """处理单个文档"""
        text_chunks = self._split_text(document.content)
        
        doc_chunks = []
        for i, text in enumerate(text_chunks):
            # 深拷贝元数据
            meta = document.metadata.copy()
            meta.update({
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "processed_at": datetime.now().isoformat()
            })
            
            chunk = DocumentChunk(
                content=text,
                metadata=meta,
                doc_id=document.doc_id,
                chunk_index=i
            )
            doc_chunks.append(chunk)
        return doc_chunks

    def _split_text(self, text: str) -> List[str]:
        """递归/迭代文本分割逻辑"""
        if not text:
            return []
        
        if len(text) <= self.chunk_size:
            return [text]
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # 如果是最后一段
            if end == text_len:
                chunks.append(text[start:])
                break
                
            # 寻找最佳分割点
            split_point = self._find_best_split(text, start, end)
            
            # 如果没找到分隔符，且必须强制分割
            if split_point == -1:
                split_point = end
            
            # 由于可能出现 split_point <= start 的死循环情况（如分隔符在start之前），做保护
            if split_point <= start:
                split_point = end
                
            chunks.append(text[start:split_point])
            
            # 计算下一次开始位置（重叠）
            # 只有当 text 足够长时才通过重叠回退，否则直接接驳
            next_start = split_point - self.chunk_overlap
            
            # 确保 next_start 至少比 start 前进 1，避免死循环
            if next_start <= start:
                next_start = start + 1
            # 确保 next_start 不会小于 split_point 如果 split_point 已经是强制截断
            if next_start >= split_point: 
                 next_start = split_point 
                 
            # 修正逻辑：通常重叠是指下一块包含上一块的末尾
            # 下一块从 split_point - overlap 开始
            start = max(start + 1, split_point - self.chunk_overlap)
            
        return chunks

    def _find_best_split(self, text: str, start: int, end: int) -> int:
        """在区间 [end-window, end] 内寻找最高优先级的分割符"""
        # 搜索窗口：只在切片末尾的一段区域寻找分隔符，避免切得太短
        # 比如只在最后 25% 或 100个字符 范围内找
        search_window = max(20, int(self.chunk_size * 0.2))
        search_start = max(start, end - search_window)
        
        for sep in self.separators:
            # rfind 从右向左找
            idx = text.rfind(sep, search_start, end)
            if idx != -1:
                return idx + len(sep) # 包含分隔符
        
        return -1


class DocumentLoader:
    """通用文档加载器工厂"""
    
    @staticmethod
    def load(file_path: str) -> List[Document]:
        """根据文件扩展名加载文档，支持返回多个文档（如JSON list）"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        suffix = path.suffix.lower()
        
        if suffix == ".pdf":
            return DocumentLoader._load_pdf(path)
        elif suffix == ".json":
            return DocumentLoader._load_json(path)
        elif suffix in [".txt", ".md", ".csv", ".py", ".log"]:
            return DocumentLoader._load_text(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def _load_text(path: Path) -> List[Document]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return [Document(
                content=content,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "type": "text",
                    "extension": path.suffix
                }
            )]
        except UnicodeDecodeError:
            # Fallback for other encodings if needed
            raise ValueError(f"Failed to decode text file: {path}")

    @staticmethod
    def _load_pdf(path: Path) -> List[Document]:
        if pypdf is None:
            raise ImportError("pypdf is required to load PDF files. Please install it with `pip install pypdf`.")
        
        docs = []
        try:
            reader = pypdf.PdfReader(str(path))
            full_text = []
            
            # 策略1：将每一页作为一个独立文档（保留页码信息），或者合并
            # 这里选择合并，但在 metadata 中记录页数，如果需要按页切分可修改此处
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text.append(text)
            
            # 合并为一个长文档，由 Processor 负责切分
            joined_text = "\n\n".join(full_text)
            
            docs.append(Document(
                content=joined_text,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "type": "pdf",
                    "total_pages": len(reader.pages)
                }
            ))
            return docs
        except Exception as e:
            raise RuntimeError(f"Error loading PDF {path}: {e}")

    @staticmethod
    def _load_json(path: Path) -> List[Document]:
        """
        加载JSON文件。
        如果 JSON 是 list，则每个元素视为一个文档。
        如果是 dict，则视为一个文档。
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            docs = []
            base_meta = {
                "source": str(path),
                "file_name": path.name,
                "type": "json"
            }
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    content = json.dumps(item, ensure_ascii=False, indent=2)
                    meta = base_meta.copy()
                    meta["index"] = i
                    # 尝试从 item 中提取更多元数据，如 id, title
                    if isinstance(item, dict):
                        if "id" in item: meta["original_id"] = str(item["id"])
                        if "title" in item: meta["title"] = str(item["title"])
                        
                    docs.append(Document(content=content, metadata=meta))
            elif isinstance(data, dict):
                content = json.dumps(data, ensure_ascii=False, indent=2)
                docs.append(Document(content=content, metadata=base_meta))
            else:
                # 简单类型
                docs.append(Document(content=str(data), metadata=base_meta))
                
            return docs
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {path}")

