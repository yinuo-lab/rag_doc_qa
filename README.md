# Local RAG Doc QA

一个基于 FastAPI、Ollama 和本地 embedding/LLM 的中文 RAG 文档问答系统，支持本地文档索引同步、检索增强生成和来源返回。

## Features

支持 txt/md 文档加载
支持递归扫描子目录
支持 doc_registry.json + chunks.jsonl 本地索引同步
支持本地 embedding 检索
支持本地 LLM 生成答案
返回 answer + sources

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

## ##系统流程

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

ollama run qwen3:8b
ollama pull qwen3-embedding:0.6b-fp16
python -m uvicorn app.main:app



## 运行测试：

## 1.数据库中没有的问题

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

## 2.数据库中有的问题

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

## 3、中文叙事类型
{"query": "沈砚去旧书店是为了什么？"}
{
  "answer": "沈砚去旧书店是为了确认一件事：三年前寄来的那封没有署名的信，里面提到的“南窗第三层，左数第二本”，到底是不是一句故意留下来的谜语。",
  "sources": [
    {
      "source": "novel_excerpt.txt",
      "doc_id": "misc/novel_excerpt.txt",
      "chunk_id": 1
    }
  ]
}
## 现状

本地模型接入
读取复数文档
本地 embedding 检索
索引同步
中文问答
## 未来工作
 Docker
 更强 reranker

## 关于这个项目

这个项目的目标不是简单调用大模型 API，而是实现一个具备基本工程结构的本地 RAG 系统，包括：

- 本地文档管理
- 本地索引同步
- 检索与重排
- 本地模型接入
- API 服务封装

它更接近一个可以继续扩展的 AI 应用工程项目，而不是单纯的实验脚本。


---
