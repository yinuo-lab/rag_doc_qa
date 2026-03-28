from pathlib import Path

from app.schemas.rag import Document

def load_documents(data_dir: str) -> list[Document]:
    base_path = Path(data_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"目录不存在: {data_dir}")

    documents = []

    for file_path in base_path.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in {".txt", ".md"}:
            continue

        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue
        document=Document(
            source=file_path.name,
            text=text
        )
        documents.append(
            document
        )

    return documents