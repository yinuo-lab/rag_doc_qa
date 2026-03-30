# Local RAG Doc QA

一个基于 FastAPI 和本地大模型的 RAG 文档问答系统。  
系统支持本地文档加载、文本切分、向量检索、候选重排与来源返回，并通过 `/ask` 接口提供问答服务。

## Features

- 支持本地 `txt/md` 文档加载
- 支持文本 chunk 切分与 overlap
- 支持本地 embedding 与 in-memory 向量检索
- 支持候选 chunk rerank
- 支持基于检索结果构造 prompt
- 支持本地大模型生成回答
- 支持 FastAPI `/ask` 接口
- 返回 `answer + sources`

## Project Structure

```text
app/
├─ api/
│  └─ routes/
│     └─ ask.py
├─ core/
│  └─ pipeline.py
├─ schemas/
│  └─ rag.py
├─ services/
│  ├─ loader.py
│  ├─ chunker.py
│  ├─ embedder.py
│  ├─ vector_store.py
│  ├─ retriever.py
│  ├─ reranker.py
│  ├─ prompt_builder.py
│  ├─ llm.py
│  └─ rag_pipeline.py
├─ data/
│  └─ ...
└─ main.py
```

## ##System Flow

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

## 技术栈

Python
FastAPI
Ollama
Local LLM
RAG
In-memory vector store
Pydantic

## 运行

1. Create and activate virtual environment
python -m venv .venv
2. Install dependencies
python -m pip install -r requirements.txt
3. Start Ollama and prepare a local model
ollama run llama3

## 或其他本地模型，例如：

ollama run qwen2.5:3b
4. Start FastAPI
python -m uvicorn app.main:app --reload
5. Open docs
http://127.0.0.1:8000/docs
API Example

## 运行测试：
{
  "query": "谁是美国总统?"
}

{
  "answer": "不知道。提供的资料中没有关于美国总统的信息。",
  "sources": [
    {
      "source": "doc1.txt",
      "doc_id": "doc1.txt",
      "chunk_id": 0
    },
    {
      "source": "doc2.md",
      "doc_id": "doc2.md",
      "chunk_id": 0
    }
  ]
}
{
  "query": "Fastapi支持什么?"
}
{
  "answer": "FastAPI支持依赖注入、异步编程和自动文档。",
  "sources": [
    {
      "source": "doc1.txt",
      "doc_id": "doc1.txt",
      "chunk_id": 0
    },
    {
      "source": "doc2.md",
      "doc_id": "doc2.md",
      "chunk_id": 0
    }
  ]
}


## 现状

 文档读取
 chunk 切分
 embedding
 本地向量检索
 rerank
 FastAPI /ask
 返回 sources
 本地持久化索引
 更强的 reranker
 更完整的 README 演示截图

## 未来工作

增加 chunk / index 本地持久化
支持文档增量更新
优化 reranking 逻辑
支持更多文档格式
增加 Docker 部署

## 关于这个项目

这个项目的目标不是简单调用大模型 API，而是实现一个具备基本工程结构的本地 RAG 系统，包括：

- 本地文档管理
- 本地索引同步
- 检索与重排
- 本地模型接入
- API 服务封装

它更接近一个可以继续扩展的 AI 应用工程项目，而不是单纯的实验脚本。


---
