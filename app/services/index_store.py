import json
from pathlib import Path
import hashlib

from sqlalchemy import false

from app.schemas.rag import DocRegistryItem, Chunk
from app.services.chunker import split_documents
from app.services.loader import load_documents


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

import json
from pathlib import Path

from app.schemas.rag import DocRegistryItem


def load_doc_registry(index_dir:str)->dict[str,DocRegistryItem]:
    path=Path(index_dir)/'doc_registry.json'
    if not path.exists():
        return {}
    Registrys={}
    with path.open('r',encoding="utf-8")as f:
        temp=json.load(f)
    for chunk,item in  temp.items():
        Registrys[chunk]=DocRegistryItem(**item)
    return Registrys

def save_doc_registry(index_dir: str, doc_registry_items: dict[str, DocRegistryItem]) :
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    registry_path = index_path / "doc_registry.json"

    raw_data = {}

    for doc_id, registry_item in doc_registry_items.items():
        raw_data[doc_id] = registry_item.model_dump()

    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

def load_chunk(index_dir:str)->list[Chunk]:
    path=Path(index_dir)/'chunks.jsonl'
    if not path.exists():
        return []
    Chunks=[]
    with path.open('r',encoding="utf-8")as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            Chunks.append(Chunk(**json.loads(line)))
    return Chunks

def save_chunk(index_dir:str,Chunks:list[Chunk]):
    path=Path(index_dir)
    if not path.exists():
        path.mkdir(parents=True)
    path = Path(index_dir) / 'chunks.jsonl'

    with path.open("w",encoding='utf-8')as f:
        for chunk in Chunks:
            f.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")


def sync_index():
    path="app/data"
    index_dir = "app/data/index"
    docRegistryItems=load_doc_registry(index_dir)
    documents=load_documents(path)
    old_chunks=load_chunk(index_dir)
    change=[]
    delete=[]
    new=[]
    pasted={}
    for document in documents:
        pasted[document.doc_id]=1
        if not document:
            continue
        content_hash=compute_content_hash(document.text)
        if document.doc_id not in docRegistryItems:
            new.append(document)
            continue
        if docRegistryItems[document.doc_id].content_hash==content_hash:
            continue
        change.append(document)
    new_registry={}
    for registry in docRegistryItems.values():
        if registry.doc_id not in pasted:
            delete.append(registry.doc_id)
        else:
            new_registry[registry.doc_id]=registry
    docRegistryItems=new_registry
    new_chunk = []
    for registry in delete:
        old_chunks=[chunk for chunk in old_chunks if chunk.doc_id!=registry]
    doc_registry = docRegistryItems
    for document in change:
        old_chunks=[chunk for chunk in old_chunks if chunk.doc_id!=document.doc_id]
        new_chunk = split_documents([document])
        for chunk in new_chunk:
            old_chunks.append(chunk)
        content_hash = compute_content_hash(document.text)
        doc_registry[document.doc_id]=DocRegistryItem(
                doc_id=document.doc_id,
                content_hash=content_hash,
                source=document.source,
                chunk_count=len(new_chunk)
            )

    for document in new:
        new_chunk = split_documents([document])
        content_hash=compute_content_hash(document.text)
        for chunk in new_chunk:
            old_chunks.append(chunk)
        doc_registry[document.doc_id]=DocRegistryItem(
                doc_id=document.doc_id,
                content_hash=content_hash,
                source=document.source,
                chunk_count=len(new_chunk)
            )

    save_doc_registry(index_dir,doc_registry)
    save_chunk(index_dir,old_chunks)
