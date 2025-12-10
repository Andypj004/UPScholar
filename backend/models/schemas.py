from pydantic import BaseModel
from typing import List, Optional

class SearchQuery(BaseModel):
    query: str
    method: str = "tfidf"  # "tfidf" o "embedding"
    top_k: int = 10
    embedding_model: Optional[str] = "all-MiniLM-L6-v2"  # Modelo a usar para embeddings

class Document(BaseModel):
    id: int
    title: str
    keywords: str
    abstract: str
    score: float
    similar_docs: Optional[List['SimilarDocument']] = []

class SimilarDocument(BaseModel):
    id: int
    title: str
    score: float

class SearchResponse(BaseModel):
    query: str
    method: str
    results: List[Document]
    total_results: int
    embedding_model: Optional[str] = None  # Modelo usado para embeddings