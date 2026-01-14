# RAG 系统实施计划 (Implementation Plan)

## 目标
集成检索增强生成 (RAG) 系统到 `FoodPlanGPT` 项目中。这将允许 `SmartMenuAgent` 查询存储在向量数据库 (Milvus) 中的外部领域知识（例如食谱 PDF、营养指南），从而提供更准确和信息丰富的回答。

## 当前状态
- **Agent**: `SmartMenuAgent` 已存在，并具备 `EpisodicMemory` (Milvus) 和 `SemanticMemory` (Neo4j)。
- **基础设施**: 
    - `MilvusConnectionManager` 已就绪 (`app/utils/milvus_store.py`)。
    - `get_text_embedder` 已就绪 (`app/model/embedder.py`)。
- **待办**: `app/services/pdf.py` 目前为空。

## 建议架构

1.  **文档加载器与分割器**:
    -   模块: `app/services/pdf.py` (稍后可重构为 `document_loader.py`)。
    -   功能: 从 PDF 文件中提取文本；将文本分割成可重叠的块 (Chunks)。
2.  **知识服务**:
    -   模块: `app/services/knowledge_service.py` (新建)。
    -   功能: 
        -   管理 Milvus 集合 `knowledge_base`。
        -   编排流程: 加载 PDF -> 分割 -> 嵌入 (Embed) -> 索引。
        -   检索流程: 查询 -> 嵌入 -> 向量搜索 -> 重排序 (可选) -> 返回上下文。
3.  **Agent 集成**:
    -   模块: `app/agent/smart_menu_agent.py`。
    -   功能: 在调用 LLM 之前，将检索到的上下文注入到 Prompt 中。

## 分步实施阶段

### 阶段 1：多格式文档处理（加载与分块）
**目标**: 实现读取 PDF、JSON 及文本文件，并将其分割为可嵌入的片段。

- [ ] **依赖检查**: 确认 `pypdf` 是否可用。
- [ ] **重构/创建 `app/services/document_loader.py`**:
    -   定义统一的 `Document` 数据结构 (content, metadata)。
    -   **PDFLoader**: 继承并实现 PDF 读取 (使用 `app/services/pdf.py` 或直接在此实现)。
    -   **JSONLoader**: 处理 JSON 数据。
        -   *策略*: 支持列表及对象。对于列表，每项作为一个文档；对于复杂对象，转换为字符串或提取特定字段。
    -   **TextLoader**: 处理 `.txt` 和 `.md` 文件。
    -   **Splitter**: 实现通用分块器 (Chunker)，支持按字符数或分隔符分块。

### 阶段 2：知识服务（向量库交互）
**目标**: 使用 Milvus 存储和检索知识。

- [ ] **创建 `app/services/knowledge_service.py`**:
    -   初始化 `MilvusConnectionManager`，集合名称设为 `knowledge_base`。
    -   实现 `add_document(file_path)`:
        1.  根据文件扩展名自动选择 Loader。
        2.  加载并分割文本。
        3.  调用 `embedder.encode()` 获取向量。
        4.  存入 Milvus，附带元数据 (来源, 类型,页码/Key)。
    -   实现 `search_knowledge(query, top_k=3)`:
        1.  嵌入查询文本。
        2.  搜索 Milvus。
        3.  格式化结果为字符串。

### 阶段 3：Agent 集成
**目标**: 将知识服务连接到 SmartMenuAgent。

- [ ] **修改 `app/agent/smart_menu_agent.py`**:
    -   在 `__init__` 中初始化 `self.knowledge_service`。
    -   更新 `step()` 或 `run()` 方法:
        -   基于用户输入执行搜索（关键词或向量）。
        -   将结果追加到 `[Reference Knowledge]` 下的系统提示词或消息历史中。
    -   (可选) CLI 更新: 添加命令以摄入文档 (支持通配符或目录扫描)。

### 阶段 4：验证
- [ ] **测试摄入**: 
    -   摄入 PDF 食谱。
    -   摄入 JSON 格式的营养数据。
- [ ] **测试检索**: 混合查询不同来源的数据，验证 Agent 响应。

## 需要用户确认的事项
- **集合名称**: 默认为 Milvus 中的 `knowledge_base` (不同格式混存)。
- **JSON 处理策略**: 默认会将 JSON 对象转为文本字符串进行嵌入。如果有特定的 JSON 结构（如 {"name": "菜名", "steps": "..."}），可能需要针对性解析。
- **库**: 将使用标准库 + `pypdf`。

---
**下一步**: 获得批准后，我将从 **阶段 1 (创建 document_loader.py)** 开始。
