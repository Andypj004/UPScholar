"""
Multiple embeddings models manager
Allows dynamic switching between different BERT models
"""

import numpy as np
import os
from typing import Dict, Optional
from .embedding_search import EmbeddingSearchEngine

class EmbeddingManager:
    def __init__(self, abstracts: list):
        """
        Initializes the embeddings manager
        
        Args:
            abstracts: List of document abstracts
        """
        self.abstracts = abstracts
        self.engines: Dict[str, EmbeddingSearchEngine] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Available models
        self.available_models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "bert-base-nli-mean-tokens"
        ]
        
        print("EmbeddingManager initialized")
        print(f"Available models: {', '.join(self.available_models)}")
    
    def get_engine(self, model_name: str) -> EmbeddingSearchEngine:
        """
        Gets or creates an embeddings engine for the specified model
        
        Args:
            model_name: Model name
            
        Returns:
            Configured EmbeddingSearchEngine
        """
        if model_name not in self.available_models:
            print(f"Model {model_name} not supported, using all-MiniLM-L6-v2")
            model_name = "all-MiniLM-L6-v2"
        
        # If engine already exists, return it
        if model_name in self.engines:
            print(f"Using existing engine for {model_name}")
            return self.engines[model_name]
        
        # Create new engine
        print(f"Creating new engine for {model_name}...")
        engine = EmbeddingSearchEngine(self.abstracts, model_name=model_name)
        
        # Try to load embeddings from cache
        embeddings_path = f"../data/embeddings_{model_name.replace('/', '_')}.npy"
        
        if os.path.exists(embeddings_path):
            print(f"Loading embeddings from {embeddings_path}...")
            try:
                embeddings = np.load(embeddings_path)
                engine.load_embeddings(embeddings)
                self.embeddings_cache[model_name] = embeddings
                print(f"✓ Embeddings loaded for {model_name}")
            except Exception as e:
                print(f"Error loading embeddings: {e}")
                print("Generating new embeddings...")
                self._generate_and_save_embeddings(engine, model_name, embeddings_path)
        else:
            print(f"Embeddings file not found: {embeddings_path}")
            print("Generating embeddings...")
            self._generate_and_save_embeddings(engine, model_name, embeddings_path)
        
        # Save engine in cache
        self.engines[model_name] = engine
        
        return engine
    
    def _generate_and_save_embeddings(self, engine: EmbeddingSearchEngine, 
                                     model_name: str, save_path: str):
        """
        Generates and saves embeddings for a model
        
        Args:
            engine: Embeddings engine
            model_name: Model name
            save_path: Path where to save
        """
        try:
            engine.generate_embeddings(batch_size=32, show_progress=True)
            
            # Save embeddings
            np.save(save_path, engine.embeddings)
            self.embeddings_cache[model_name] = engine.embeddings
            
            print(f"✓ Embeddings generated and saved in {save_path}")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
    
    def search(self, query: str, model_name: str, top_k: int = 10):
        """
        Performs search with the specified model
        
        Args:
            query: Search query
            model_name: Model to use
            top_k: Number of results
            
        Returns:
            List of tuples (index, score)
        """
        engine = self.get_engine(model_name)
        return engine.search(query, top_k=top_k)
    
    def get_similar_documents(self, doc_index: int, model_name: str, 
                            top_k: int = 3, exclude_indices: set = None):
        """
        Gets similar documents using the specified model
        
        Args:
            doc_index: Document index
            model_name: Model to use
            top_k: Number of results
            exclude_indices: Indices to exclude
            
        Returns:
            List of tuples (index, score)
        """
        engine = self.get_engine(model_name)
        return engine.get_similar_documents(doc_index, top_k, exclude_indices)
    
    def preload_model(self, model_name: str):
        """
        Pre-loads a model to have it ready
        
        Args:
            model_name: Model name to pre-load
        """
        try:
            self.get_engine(model_name)
            print(f"✓ Model {model_name} pre-loaded successfully")
        except Exception as e:
            print(f"Error pre-loading model {model_name}: {e}")
    
    def get_available_models(self):
        """Returns list of available models"""
        return self.available_models
    
    def get_loaded_models(self):
        """Returns list of models already loaded in memory"""
        return list(self.engines.keys())