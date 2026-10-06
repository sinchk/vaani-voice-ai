import os
import chromadb
from vaani.app.rag.embeddings import EmbeddingModel

def ingest_knowledge_base(data_dir="vaani/data/knowledge_base", persist_dir="vaani/chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="kb_collection")
    embedder = EmbeddingModel()

    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
                # Simple chunking by paragraph/newlines for this prototype
                chunks = [c.strip() for c in content.split("\n") if len(c.strip()) > 20]
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"source": filename})
                    ids.append(f"{filename}_{i}")

    if documents:
        embeddings = embedder.embed_documents(documents)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Ingested {len(documents)} chunks from {data_dir}")

if __name__ == "__main__":
    ingest_knowledge_base()
