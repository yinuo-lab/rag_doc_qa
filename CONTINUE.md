# Local RAG Doc QA — 项目技能说明

## 项目概括

基于 FastAPI + Ollama + 本地 embedding/LLM 的中文 RAG 文档问答系统。

## 技术栈

- Python / FastAPI / Pydantic
- Ollama（本地大模型运行器）
- qwen3:8b（生成模型）
- qwen3-embedding:0.6b-fp16（嵌入模型）
- 内存向量存储（InMemoryVectorStore）
- NumPy（余弦相似度计算）

## 项目结构

```
app/
├── api/routes/ask.py         # API 路由（/ask, /health, /echo）
├── core/pipeline.py          # 启动入口：构建 pipeline 单例
├── core/config.py            # 配置
├── schemas/rag.py            # Pydantic 数据模型
├── services/
│   ├── loader.py             # 从 data/ 目录递归加载 txt/md
│   ├── chunker.py            # 文本切分（固定大小 + overlap）
│   ├── embedder.py           # 调用 Ollama 生成向量
│   ├── vector_store.py       # 内存向量存储 + 余弦相似度检索
│   ├── retriever.py          # 检索器
│   ├── reranker.py           # 重排器
│   ├── prompt_builder.py     # Prompt 构造
│   ├── llm.py               # 调 Ollama LLM 生成回答
│   ├── rag_pipeline.py      # RAG 主流程串联
│   └── index_store.py       # 本地索引同步（doc_registry.json + chunks.jsonl）
└── data/                     # 测试文档目录
```

## RAG 主流程

```
文档 → loader → chunker → embedder → vector_store
查询 → retriever → reranker → prompt_builder → llm → answer + sources
```

## 当前已知的可改进点

1. reranker 的重排实现有 bug——不是在算语义相似度，而是在数向量中相等且非零的维度数
2. 没有阈值过滤——检索结果不管多不相关都会被送进 prompt
3. 没有优雅降级——检索不到相关内容时不会主动告知用户
4. prompt 里没有处理多文档矛盾——如果多个文档说法不同，模型可能会强行融合
5. 没有流式输出——用户要等全部生成完才能看到回答
6. 没有对话记忆——每轮问答互相独立
7. Embedding 没有持久化——重启服务要重新算所有向量
8. 检索策略单一——只有向量检索，没有 BM25 关键词检索
9. 没有质量评估体系——无法量化 RAG 效果好坏

## 可用的 Skills（位于 ~/.continue/skills/agent-skills/skills/）

- **powershell-safe-invocation**: Windows PowerShell 安全调用规范（原生命令参数、路径、引号、编码）
- **bilibili-page-reader**: 读取 Bilibili 视频页，获取字幕、弹幕、评论

## 运行方式

```bash
ollama run qwen3:8b
ollama pull qwen3-embedding:0.6b-fp16
python -m uvicorn app.main:app
```
