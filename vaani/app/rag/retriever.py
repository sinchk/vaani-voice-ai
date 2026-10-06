import chromadb
from vaani.app.rag.embeddings import EmbeddingModel
from vaani.app.rag.ingest import ingest_knowledge_base

class KBRetriever:
    def __init__(self, persist_dir="vaani/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="kb_collection")
        if self.collection.count() == 0:
            print("Chroma collection empty. Ingesting knowledge base...")
            ingest_knowledge_base(persist_dir=persist_dir)
        self.embedder = EmbeddingModel()

    def retrieve(self, query, top_k=3, min_score=0.6):
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_docs = []
        for i in range(len(results["documents"][0])):
            # Distance in Chroma is often L2, score = 1 / (1 + distance) or similar
            # For simplicity, we'll use a placeholder logic or assume similarity
            distance = results["distances"][0][i]
            confidence = 1 - (distance / 2) # Heuristic for cosine-like distance
            
            if confidence >= min_score:
                retrieved_docs.append({
                    "content": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "confidence": round(confidence, 2)
                })
        
        return retrieved_docs
