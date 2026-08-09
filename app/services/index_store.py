# app/services/index_store.py
# 简化版：逻辑与原版完全一致，只把复杂语法改成最直白的写法，方便学习

import hashlib
import json
from pathlib import Path

from app.schemas.rag import Chunk, DocRegistryItem
from app.services.chunker import split_documents
from app.services.loader import load_documents


def compute_content_hash(text: str) -> str:
    """对文档内容算 SHA256 指纹。内容变一个字符，指纹就完全不同。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_doc_registry(index_dir: str) -> dict:
    """读取登记簿：doc_id -> DocRegistryItem。文件不存在或为空时返回空字典。"""
    path = Path(index_dir) / "doc_registry.json"
    if not path.exists():
        return {}

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}

    data = json.loads(content)
    result = {}
    for doc_id, item in data.items():
        registry = DocRegistryItem(
            doc_id=item["doc_id"],
            source=item["source"],
            content_hash=item["content_hash"],
            chunk_count=item["chunk_count"],
        )
        result[doc_id] = registry
    return result


def save_doc_registry(index_dir: str, items: dict) -> None:
    """把登记簿写回磁盘。目录不存在会自动创建。"""
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    raw = {}
    for doc_id, item in items.items():
        raw[doc_id] = {
            "doc_id": item.doc_id,
            "source": item.source,
            "content_hash": item.content_hash,
            "chunk_count": item.chunk_count,
        }

    path = index_path / "doc_registry.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)


def load_chunk(index_dir: str) -> list:
    """读取 chunks.jsonl：每行一个 chunk 的 JSON。"""
    path = Path(index_dir) / "chunks.jsonl"
    if not path.exists():
        return []

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item:
                continue
            chunk = Chunk(
                doc_id=item["doc_id"],
                source=item["source"],
                chunk_id=item["chunk_id"],
                text=item["text"],
            )
            chunks.append(chunk)
    return chunks


def save_chunk(index_dir: str, chunks: list) -> None:
    """把 chunks 写回 chunks.jsonl，一行一个。"""
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    path = index_path / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            line = json.dumps(
                {
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                },
                ensure_ascii=False,
            )
            f.write(line + "\n")


def sync_index(data_dir: str, index_dir: str = "app/data/index") -> list:
    """增量同步索引：对比登记簿和磁盘上的文档，只处理新增/修改/删除。"""
    old_registry = load_doc_registry(index_dir)
    documents = load_documents(data_dir)
    chunks = load_chunk(index_dir)

    new_docs = []      # 新增：登记簿里没有
    changed_docs = []  # 修改：哈希变了
    deleted_ids = []   # 删除：登记簿有、磁盘没有

    # 第一步：给每个文档分类
    on_disk = set()
    for document in documents:
        on_disk.add(document.doc_id)
        content_hash = compute_content_hash(document.text)

        if document.doc_id not in old_registry:
            new_docs.append(document)
        elif old_registry[document.doc_id].content_hash != content_hash:
            changed_docs.append(document)
        # 哈希相同：文档没变，什么都不做，直接复用旧 chunk

    # 第二步：找出被删除的文档
    for doc_id in old_registry:
        if doc_id not in on_disk:
            deleted_ids.append(doc_id)

    # 第三步：处理删除——移除它的所有 chunk，并从登记簿删除
    for doc_id in deleted_ids:
        remaining = []
        for c in chunks:
            if c.doc_id != doc_id:
                remaining.append(c)
        chunks = remaining
        del old_registry[doc_id]

    # 第四步：处理修改——删旧块、重新切分、更新登记簿
    for document in changed_docs:
        remaining = []
        for c in chunks:
            if c.doc_id != document.doc_id:
                remaining.append(c)
        chunks = remaining

        new_chunks = split_documents([document])
        chunks.extend(new_chunks)

        old_registry[document.doc_id] = DocRegistryItem(
            doc_id=document.doc_id,
            source=document.source,
            content_hash=compute_content_hash(document.text),
            chunk_count=len(new_chunks),
        )

    # 第五步：处理新增——切分、加入、写入登记簿
    for document in new_docs:
        new_chunks = split_documents([document])
        chunks.extend(new_chunks)

        old_registry[document.doc_id] = DocRegistryItem(
            doc_id=document.doc_id,
            source=document.source,
            content_hash=compute_content_hash(document.text),
            chunk_count=len(new_chunks),
        )

    # 第六步：持久化
    save_doc_registry(index_dir, old_registry)
    save_chunk(index_dir, chunks)
    return chunks
