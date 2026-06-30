# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 概述

这是 **LangChain Academy** —— 一门教授 LangGraph 的课程。它是上游 `langchain-ai/langchain-academy` 的一个分支，已适配为使用 **DeepSeek** 作为 LLM 提供商（`langchain-deepseek`、`ChatDeepSeek`），而非 OpenAI。仓库包含 7 个模块（0–6）的 Jupyter Notebook 以及配套的 LangGraph Studio 图，内容从基础概念逐步深入到生产部署。

## 常用命令

```bash
# 安装依赖（需要 Python 3.11–3.13）
pip install -r requirements.txt

# 启动 Jupyter 运行 notebook
jupyter notebook

# 本地运行 LangGraph Studio（在任意 module-x/studio/ 目录下）
# 首先复制 .env.example 并设置你的 API 密钥：
cp module-X/studio/.env.example module-X/studio/.env
# 然后启动：
cd module-X/studio && langgraph dev

# 对于模块 6 的部署，使用 Docker Compose：
cd module-6/deployment && docker compose -f docker-compose-example.yml up
```

## 架构

### 模块递进

| 模块 | 主题 | 内容 |
|------|------|------|
| 0 | 环境搭建与基础 | `basics.ipynb` |
| 1 | 图基础 | 简单图、链、Agent、路由、工具调用、Agent 记忆、部署 |
| 2 | 状态管理 | 状态 Schema、Reducer、多 Schema、聊天机器人摘要、外部记忆、消息裁剪 |
| 3 | 人机协同 | 断点、动态断点、状态编辑、流式中断、时间旅行 |
| 4 | 高级编排 | 并行化、子图、Map-Reduce、研究助手（多 Agent） |
| 5 | 长期记忆 | InMemoryStore、Trustcall 提取、Profile/Collection Schema、记忆 Agent |
| 6 | 生产部署 | Assistant 创建、连接、双重消息、Docker Compose 部署 |

### Studio 图模式

每个 `module-X/studio/` 目录是一个独立的 LangGraph Studio 项目：

- **`langgraph.json`** —— 将可读的图名称映射到 `模块:变量` 引用（例如 `"agent": "./agent.py:graph"`）。编译后的图变量必须命名为 `graph`。
- **`.env.example`** —— 所需 API 密钥的模板（DeepSeek、LangSmith、Tavily）
- **`requirements.txt`** —— 模块特定的依赖
- **`*.py` 文件** —— 每个文件定义并导出一个已编译的 `graph`（即 `builder.compile()` 的结果）

图变量名 `graph` 是约定 —— `langgraph.json` 引用它，LangGraph Studio 发现它。

### 关键库及其作用

- **`langgraph`** —— 核心图框架：`StateGraph`、`START`、`END`、`Send`、`ToolNode`、`tools_condition`、`MemorySaver`
- **`langgraph.store.base.BaseStore` / `InMemoryStore`** —— 跨对话的持久化键值记忆（模块 5–6 大量使用）
- **`langchain-deepseek`** —— `ChatDeepSeek(model="deepseek-chat")` 是所有模块使用的 LLM
- **`trustcall`** —— `create_extractor()` 用于结构化记忆提取，支持插入/更新/补丁语义
- **`langchain-tavily`** —— `TavilySearch` 用于网络搜索（模块 4）
- **`langgraph-cli[inmem]`** —— `langgraph dev` 命令，用于本地启动 Studio

### 本仓库中常见的 LangGraph 模式

1. **工具调用 Agent**（模块 1、3）：`assistant` 节点 → `tools_condition` → `ToolNode` → 循环回到 assistant
2. **带摘要的聊天机器人**（模块 2）：对话节点 → 条件边（消息数 > N）→ 摘要节点 → END
3. **使用 `Send()` 的并行扇出**（模块 4）：父节点返回 `[Send("子图", {...}) for item in items]`，动态生成并行子图执行
4. **子图组合**（模块 4）：将一个 `StateGraph` 编译为子图，然后通过 `builder.add_node("名称", 子图.compile())` 将其作为父图中的一个节点
5. **基于 `BaseStore` 的记忆**（模块 5–6）：`call_model` 节点从 Store 读取 → 响应 → `write_memory` 节点使用 Trustcall 提取/更新记忆 → 存回 Store。运行时配置通过从 `RunnableConfig` 中读取的 `Configuration` 数据类实现。

### 配置模式（模块 5–6）

`configuration.py` 定义了一个 `@dataclass`，包含诸如 `user_id` 等字段。它提供 `from_runnable_config(config)` 方法，从 `config["configurable"]` 字典或环境变量（字段名大写）中读取配置。图通过 `StateGraph(State, config_schema=configuration.Configuration)` 构建，节点通过 `configuration.Configuration.from_runnable_config(config)` 访问配置。

### 所需的 API 密钥

- `DEEPSEEK_API_KEY` —— 所有模块都需要
- `LANGSMITH_API_KEY` + `LANGSMITH_TRACING_V2=true` —— LangSmith 追踪
- `TAVILY_API_KEY` —— 模块 4 需要（网络搜索）

### Notebook ↔ Studio 关系

每个 Notebook 教授一个概念；对应的 `studio/` 目录包含该模块的可运行图。Notebook 是学习材料，Studio 图是交互式实践环境。修改图代码时，需要同时更新 Notebook（教学用途）和 Studio 的 `.py` 文件（运行用途）。
