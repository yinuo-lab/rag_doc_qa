from app.services.loader import load_documents
from app.services.chunker import split_documents


def main():
    documents = load_documents("app/data")
    print("=== documents ===")
    for doc in documents:
        print(doc["source"], "长度:", len(doc["text"]))

    chunks = split_documents(documents, chunk_size=80, overlap=20)
    print("\n=== chunks ===")
    for chunk in chunks:
        print(chunk)
        print("-" * 50)


if __name__ == "__main__":
    main()