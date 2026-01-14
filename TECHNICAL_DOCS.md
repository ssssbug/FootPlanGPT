# FoodPlanGPT 技术架构文档

**项目名称**: FoodPlanGPT (属于韩晗的一周食谱)  
**当前版本**: 0.1.0  
**最后更新**: 2026-01-09

---

## 1. 项目概述

**FoodPlanGPT** 是一个智能食谱推荐系统，旨在根据一周的天气情况、当季蔬菜价格以及用户的历史偏好，为用户生成个性化的一周三餐食谱。系统采用 **FastAPI** 作为后端框架，集成了基于 **LLM (大语言模型)** 的 Agent 智能体，并配备了先进的 **Memory (记忆系统)** 模块，支持情景记忆 (Episodic Memory) 和工作记忆 (Working Memory)，以实现长期个性化交互。

### 核心功能
*   **智能菜谱生成**: 结合环境数据（天气、菜价）与用户对话生成菜谱。
*   **多模态记忆系统**:
    *   **工作记忆 (Working Memory)**: 处理当前会话的上下文，支持意图识别和短期交互。
    *   **情景记忆 (Episodic Memory)**: 使用向量数据库 (Milvus) 和关系型数据库 (PostgreSQL) 存储历史交互事件，支持模式识别。
*   **Agent 交互**: 基于 ReAct 模式的智能体，支持多步推理和工具调用。

---

## 2. 系统架构

系统整体架构分为三层：**接口层 (API/CLI)**、**核心逻辑层 (Agent & Services)** 和 **基础服务层 (Memory & Storage)**。

### 架构图示 (概念)
```mermaid
graph TD
    User[用户] --> |HTTP / CLI| Interface[接入层]
    Interface --> |FastAPI| API[API 服务]
    Interface --> |Interactive| CLI[命令行工具]
    
    API --> Planner[Planner Service]
    CLI --> SMAgent[SmartMenuAgent]
    
    SMAgent --> LLM[LLM 服务]
    SMAgent --> WM[Working Memory]
    SMAgent --> EM[Episodic Memory]
    
    EM --> Milvus[向量存储 (Milvus)]
    EM --> PG[关系存储 (PostgreSQL)]
```

---

## 3. 目录与模块结构

项目采用典型的 Python 模块化结构：

```text
app/
├── main.py                 # FastAPI 应用入口
├── agent/                  # 智能体核心逻辑
│   ├── agent.py            # Agent 抽象接口
│   ├── baseAgent.py        # Agent 基类实现
│   └── smart_menu_agent.py # 核心菜谱智能体 (ReAct模式)
├── memory/                 # 记忆系统 (核心亮点)
│   ├── baseMemory.py       # 记忆基类与配置
│   ├── WorkingMemory.py    # 短期工作记忆实现
│   ├── EpisodicMemory.py   # 长期情景记忆实现
│   └── storage/            # 存储后端相关
├── api/                    # API 路由
│   └── meal.py             # 食谱相关接口
├── services/               # 业务服务
│   ├── planner.py          # 食谱规划服务 (目前为Stub)
│   ├── recipe.py           # 菜谱详情服务
│   ├── weather_service.py  # 天气服务
│   └── pdf.py              # PDF 导出服务
├── llm/                    # LLM 集成
│   └── select_llm.py       # LLM 调用封装
├── schemas/                # 数据模型 (Pydantic)
├── prompt/                 # 提示词模板
│   └── default_prompt.py   # ReAct 模板与意图识别 Prompt
└── utils/                  # 工具函数
    ├── milvus_store.py     # Milvus 连接管理
    └── text_process.py     # 文本处理与分词
```

---

## 4. 核心模块详解

### 4.1 智能体 (Agent)
位于 `app/agent/smart_menu_agent.py`。
*   **类**: `SmartMenuAgent(BaseAgent)`
*   **模式**: ReAct (Reasoning and Acting)。
*   **工作流**:
    1.  **Input**: 接收用户自然语言输入。
    2.  **Prompt Construction**: 结合 `WorkingMemory` 中的意图和历史上下文构建 Prompt。
    3.  **LLM Invoke**: 调用 LLM (如 GPT-5-mini)。
    4.  **Parse**: 解析 LLM 返回的 JSON 格式 (Thought/Action)。
    5.  **Execution**: 执行动作（Finish, Continue 等）。
    6.  **Loop**: 循环执行直到完成任务或达到 `max_steps` (默认 20)。
*   **交互**: 提供了 `cli()` 方法用于命令行直接交互测试。

### 4.2 记忆系统 (Memory System)
位于 `app/memory/`，是系统的核心亮点。

#### A. 基础架构 (`baseMemory.py`)
*   定义了 `MemoryItem` 数据结构：包含 `content` (内容), `keyword` (关键词), `importance` (重要性), `timestamp` (时间戳) 等。
*   定义了 `MemoryConfig`：管理存储路径、容量限制、衰减因子 (`decay_factor`) 等。
*   实现了重要性计算算法：基于 Jaccard 相似度计算当前记忆与历史记忆的关联度。

#### B. 情景记忆 (`EpisodicMemory.py`)
*   **用途**: 存储具体的交互事件 (Episode)，用于长期的个性化回忆。
*   **存储机制**: 双层存储。
    *   **PostgreSQL**: (通过 `PostgreSQLDocumentStore`) 作为权威文档存储，保存完整的事件细节。
    *   **Milvus**: 向量数据库，存储 Embedding 后的向量，用于语义搜索 (Semantic Search)。
*   **功能**:
    *   `add`: 添加新事件，同时更新向量库和关系库。
    *   `retrieve`: 基于语义相似度检索相关历史事件。
    *   `patterns`: 具备模式识别缓存 (`patterns_cache`)，用于未来分析用户习惯。

#### C. 工作记忆 (`WorkingMemory.py`)
*   **用途**: 维护当前会话的上下文，类似于计算机的 RAM。
*   **特性**:
    *   纯内存存储，速度快。
    *   当容量不足时，根据重要性和时间（TTL）进行遗忘或归档。
    *   用于 Agent 构建每一轮对话的 Prompt Context。

### 4.3 Utils 与工具
*   **MilvusStore**: `app/utils/milvus_store.py` 封装了 `MilvusConnectionManager`，实现了单例模式管理 Milvus 数据库连接。
*   **TextProcess**: `app/utils/text_process.py` 使用 `jieba` 进行中文分词，配合 `TfidfVectorizer` (sklearn) 提供基础的文本特征处理能力。

---

## 5. 技术栈

| 类别 | 技术/库 | 说明 |
| :--- | :--- | :--- |
| **语言** | Python 3.10+ | 核心开发语言 |
| **Web 框架** | FastAPI | 高性能 API 服务, Pydantic 数据验证 |
| **服务器** | Uvicorn | ASGI 服务器 |
| **数据库 (向量)** | Milvus | `pymilvus` 客户端，用于向量存储与检索 |
| **数据库 (关系)** | PostgreSQL | (规划中/部分实现) 用于存储结构化数据 |
| **NLP & AI** | Scikit-learn | TF-IDF 文本特征提取 |
| **NLP & AI** | Jieba | 中文分词 |
| **LLM 接口** | OpenAI 兼容接口 | 通过 `chatanywhere` 等 Provider 调用 GPT 模型 |

---

## 6. 环境配置

系统依赖以下环境变量 (参考 `.env` 文件)：
```bash
# LLM 配置
OPENAI_API_KEY=sk-xxxx
API_BASE_URL=https://api.chatanywhere.tech/v1

# Milvus 向量数据库配置
MILVUS_URL=https://in03-xxxx.aws.zillizcloud.com
MILVUS_API_KEY=xxxx

# 数据库路径 (可选)
STORAGE_PATH=./.memory_data
```

## 7. 待优化项与未来规划 (Roadmap)

### 7.1 即期优化 (Fixes)
1.  **API 对接**: `app/api/meal.py` 目前返回的是硬编码数据 (`planner.py` 中的静态字典)，需要对接 `SmartMenuAgent` 来动态生成。
2.  **服务实现**: `weather_service.py` 尚未实现具体的天气 API 调用逻辑。
3.  **记忆持久化**: `EpisodicMemory` 中的 PostgreSQL 连接部分目前使用了本地 SQLite 或文件模拟 (`memory.db`)，需要确认生产环境的 PG 连接。

### 7.2 核心功能增强：情景记忆集成 (Episodic Memory Integration)
目前情景记忆模块相对独立，未来将深度集成到 Agent 的生命周期中，实现真正的“长期记忆”能力。

**实施方案**:
1.  **初始化注入**: 在 `SmartMenuAgent` 初始化时加载 `EpisodicMemory` 实例。
2.  **检索增强 (RAG)**:
    *   在每一轮对话 Step 开始前，使用当前用户的 Input Embeddings 在 Milvus 中检索 Top-K 相似的历史 Episode。
    *   将检索到的历史上下文 (如：“用户上次说不喜欢吃香菜”) 拼接到 Prompt 的 Context 部分。
3.  **记忆写入**:
    *   在对话结束 (Finish) 时，将本次会话的摘要、最终生成的食谱以及用户的反馈封装为 `Episode` 对象。
    *   调用 `add()` 方法同时写入 Vector Store (Milvus) 和 Document Store (PostgreSQL)。

### 7.3 架构升级：引入图数据库 (Graph Memory)
为了处理食材、营养、用户偏好之间复杂的结构化关系，计划引入图数据库 (Neo4j) 构建 **语义记忆 (Semantic Memory)**。

**技术选型**:
*   **Database**: Neo4j (推荐) 或 NebulaGraph。
*   **Driver**: `neo4j` Python Driver。

**图谱设计 (Schema)**:
*   **节点 (Nodes)**:
    *   `User`: 用户实体 (属性: 姓名, 年龄, 基础代谢率)
    *   `Ingredient`: 食材 (属性: 季节, 卡路里/100g, 嘌呤含量)
    *   `Dish`: 菜品 (属性: 菜系, 烹饪时长)
    *   `Tag`: 标签 (属性: "辣", "清淡", "低脂")
*   **关系 (Edges)**:
    *   `(:User)-[:LIKES]->(:Dish)`: 用户偏好
    *   `(:User)-[:ALLERGIC_TO]->(:Ingredient)`: 过敏原
    *   `(:Dish)-[:CONTAINS]->(:Ingredient)`: 配方构成
    *   `(:Dish)-[:HAS_TAG]->(:Tag)`: 属性归类

**应用场景**:
1.  **精准排除**: 当用户要求“不要辣”时，通过图谱 `MATCH (d:Dish)-[:HAS_TAG]->(t:Tag {name:'辣'}) DETACH DELETE d` (逻辑删除) 进行精准过滤。
2.  **关联推荐**: 基于“同一种食材的不同做法”或“营养互补”进行推理推荐 (GraphRAG)。
