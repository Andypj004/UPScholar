import numpy as np
from typing import List, Tuple
import os

class EmbeddingSearchEngine:
    def __init__(self, abstracts: List[str], model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the search engine with embeddings using sentence-transformers
        
        Args:
            abstracts: List of document abstracts
            model_name: Model name to use. Recommended options:
                - "all-MiniLM-L6-v2" (fast, 384 dims) - RECOMMENDED
                - "all-mpnet-base-v2" (better quality, 768 dims)
                - "paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
                - "bert-base-nli-mean-tokens" (classic BERT)
        """
        self.abstracts = abstracts
        self.embeddings = None
        self.model_name = model_name
        self.model = None
        
        # Initialize model
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            print(f"✓ Model loaded correctly")
        except ImportError:
            print("ERROR: sentence-transformers is not installed")
            print("Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_embeddings(self, batch_size: int = 32, show_progress: bool = True):
        """
        Generates embeddings for all abstracts using sentence-transformers
        
        Args:
            batch_size: Batch size for processing
            show_progress: Show progress bar
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        print(f"\nGenerating embeddings for {len(self.abstracts)} documents...")
        print(f"Model: {self.model_name}")
        print(f"Batch size: {batch_size}\n")
        
        try:
            # Generate embeddings in batches
            self.embeddings = self.model.encode(
                self.abstracts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            print(f"\n✓ Embeddings generated successfully")
            print(f"  Shape: {self.embeddings.shape}")
            print(f"  Dimension: {self.embeddings.shape[1]}")
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
        
    def load_embeddings(self, embeddings: np.ndarray):
        """Loads pre-generated embeddings"""
        self.embeddings = embeddings
        print(f"✓ Embeddings loaded: shape {embeddings.shape}")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Searches for documents similar to the query using embeddings
        
        Args:
            query: Query text
            top_k: Number of documents to return
            
        Returns:
            List of tuples (document_index, similarity)
        """
        if self.embeddings is None:
            print("ERROR: Embeddings not generated")
            raise ValueError("Embeddings not generated. Call generate_embeddings() first.")
        
        if self.model is None:
            print("ERROR: Model not available")
            raise ValueError("Model not available.")
        
        # Generate query embedding
        try:
            print(f"Generating embedding for query: '{query}'")
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            print(f"✓ Query embedding generated: shape {query_embedding.shape}")
        except Exception as e:
            print(f"ERROR generating query embedding: {e}")
            raise
        
        # Calculate similarities using vectorization (much faster)
        # Normalize vectors
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        docs_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarity with dot product
        similarities = np.dot(docs_norm, query_norm)
        
        # Create list of tuples (index, similarity)
        similitudes = [(i, float(sim)) for i, sim in enumerate(similarities)]
        
        # Sort by descending similarity
        similitudes.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✓ Top 5 similarities: {[(idx, f'{sim:.4f}') for idx, sim in similitudes[:5]]}")
        
        return similitudes[:top_k]
    
    def get_similar_documents(self, doc_index: int, top_k: int = 3, exclude_indices: set = None) -> List[Tuple[int, float]]:
        """
        Gets the most similar documents to a given document
        
        Args:
            doc_index: Document index
            top_k: Number of similar documents to return
            exclude_indices: Set of indices to exclude
        
        Returns:
            List of tuples (document_index, similarity)
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not generated.")
        
        if exclude_indices is None:
            exclude_indices = set()
        
        doc_embedding = self.embeddings[doc_index]
        
        # Normalize
        doc_norm = doc_embedding / np.linalg.norm(doc_embedding)
        docs_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(docs_norm, doc_norm)
        
        # Create list excluding specified indices
        similitudes = [
            (i, float(similarities[i])) 
            for i in range(len(similarities)) 
            if i != doc_index and i not in exclude_indices
        ]
        
        # Sort by descending similarity
        similitudes.sort(key=lambda x: x[1], reverse=True)
        
        return similitudes[:top_k]