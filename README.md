# Local RAG Doc QA

一个基于 FastAPI、Ollama 和本地 embedding/LLM 的中文 RAG 文档问答系统，支持本地文档索引同步、检索增强生成和来源返回。

## Features

支持 txt/md 文档加载
支持递归扫描子目录
支持 doc_registry.json + chunks.jsonl 本地索引同步
支持本地 embedding 检索
支持本地 LLM 生成答案
返回 answer + sources
保留普通 RAG `/ask`
支持 LangGraph + Function Calling 的 `/agent/ask`
支持模型改写查询后的多轮检索、检索次数上限、资料不足拒答和异常兜底
支持 SHA256 文档增量同步和 chunk embedding 磁盘缓存

## Project Structure

```text
app/
├─ api/
│  └─ routes/
│     ├─ ask.py
│     └─ agent.py
├─ core/
│  └─ pipeline.py
├─ schemas/
│  ├─ rag.py
│  └─ agent.py
├─ services/
│  ├─ loader.py
│  ├─ chunker.py
│  ├─ embedder.py
│  ├─ vector_store.py
│  ├─ retriever.py
│  ├─ reranker.py
│  ├─ prompt_builder.py
│  ├─ llm.py
│  ├─ rag_pipeline.py
│  ├─ retrieval_tool.py
│  ├─ embedding_cache.py
│  └─ agent_graph.py
├─ data/
│  └─ ...
└─ main.py
```

## 系统流程

documents
-> loader
-> chunker
-> embedder
-> vector_store

query
-> retriever
-> reranker
-> prompt_builder
-> llm
-> answer + sources

Agent query
-> model node
-> conditional edge
-> retrieval tool node
-> insufficient context: rewrite query and retry
-> model node
-> answer + sources

## 技术栈

Python
FastAPI
Ollama
Local LLM
RAG
In-memory vector store
Pydantic
LangGraph
Function Calling

## 运行

先确认 Ollama 已启动并安装两个模型：

```powershell
ollama list
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b-fp16
```

在项目根目录安装依赖并启动：

```powershell
cd D:\1rag_doc_qa
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

启动成功后可打开接口文档：`http://127.0.0.1:8000/docs`

### 普通 RAG 接口

`POST /ask`

```json
{
  "query": "FastAPI 支持什么？"
}
```

### Agent 接口

`POST /agent/ask`

```json
{
  "query": "RAG 项目的向量索引在重启后会怎样？",
  "max_steps": 2
}
```

Agent 响应包含 `answer`、`sources`、`steps` 和 `stop_reason`。资料不足但仍有
剩余次数时，模型必须改写查询再次调用检索工具；到达上限后程序强制拒答。

## 测试

以下测试不依赖真实 Ollama，覆盖正常检索、查询改写后二次检索、资料不足、
循环上限、检索异常，以及 embedding 增量复用：

```powershell
.\.venv\Scripts\python.exe -m app.test_agent_graph
.\.venv\Scripts\python.exe -m app.test_embedding_cache
```

预期输出：`AGENT_GRAPH_TEST=PASS`、`EMBEDDING_CACHE_TEST=PASS`

## 现状

本地模型接入
读取复数文档
本地 embedding 检索
索引同步
未变化 chunk 的 embedding 磁盘复用
中文问答
最小 LangGraph Agent
真实 Ollama Function Calling
独立 Agent 接口
轻量混合重排（向量召回分数 + 字面重合度）
## 未来工作
 Docker
 Cross-Encoder reranker
 超时、重试、鉴权、审计和正式评测集

## 实现边界

- 文档登记簿、chunks 和 embeddings 会持久化；运行时向量检索结构仍在内存中。
- 文档新增、修改、删除后只为缓存未命中的 chunks 生成 embedding。
- 当前重排是轻量混合规则，不是 Cross-Encoder。
- Agent 的查询改写由模型根据上一轮工具结果决定，最多执行请求中的 `max_steps` 次检索。

## 关于这个项目

这个项目的目标不是简单调用大模型 API，而是实现一个具备基本工程结构的本地 RAG 系统，包括：

- 本地文档管理
- 本地索引同步
- 检索与重排
- 本地模型接入
- API 服务封装

它更接近一个可以继续扩展的 AI 应用工程项目，而不是单纯的实验脚本。


---
