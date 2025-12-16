from pydantic import BaseModel
from typing import List, Optional

class SearchQuery(BaseModel):
    query: str
    method: str = "tfidf"  # "tfidf" or "embedding"
    top_k: int = 10
    embedding_model: Optional[str] = "all-MiniLM-L6-v2"  # Model to use for embeddings

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
    embedding_model: Optional[str] = None  # Model used for embeddings