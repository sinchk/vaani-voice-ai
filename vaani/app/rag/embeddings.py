import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class EmbeddingModel:
    def __init__(self, model_name="models/gemini-embedding-001"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_name = model_name

    def embed_documents(self, texts):
        return [
            genai.embed_content(model=self.model_name, content=t)["embedding"]
            for t in texts
        ]

    def embed_query(self, text):
        return genai.embed_content(model=self.model_name, content=text)["embedding"]
