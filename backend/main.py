from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import os

from models.schemas import SearchQuery, SearchResponse, Document, SimilarDocument
from services.text_processing import procesar_texto, procesar_documentos
from services.tfidf_search import TFIDFSearchEngine
from services.embedding_manager import EmbeddingManager

app = FastAPI(title="UPScholar API", version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if os.path.exists("../frontend"):
    app.mount("/static", StaticFiles(directory="../frontend", html=True), name="static")

# Global variables
tfidf_engine = None
embedding_manager = None
df_papers = None

@app.on_event("startup")
async def startup_event():
    """Initializes search engines on application startup"""
    global tfidf_engine, embedding_manager, df_papers
    
    print("\n" + "="*80)
    print("INITIALIZING UPSCHOLAR v2.0")
    print("="*80 + "\n")
    
    try:
        # Load data
        df_papers = pd.read_csv("../data/scientific_papers.csv", encoding='latin-1')
        df_papers['title'] = df_papers['title'].fillna('')
        df_papers['keywords'] = df_papers['keywords'].fillna('')
        df_papers['abstract'] = df_papers['abstract'].fillna('')
        
        titles = df_papers['title'].tolist()
        keywords = df_papers['keywords'].tolist()
        abstracts = df_papers['abstract'].tolist()
        
        print(f"✓ Loaded {len(df_papers)} documents\n")
        
        # Initialize TF-IDF engine
        print("1. Initializing TF-IDF engine...")
        tfidf_engine = TFIDFSearchEngine(titles, keywords, abstracts)
        
        # Process each component separately
        processed_titles = procesar_documentos(titles, idioma="english")
        processed_keywords = procesar_documentos(keywords, idioma="english")
        processed_abstracts = procesar_documentos(abstracts, idioma="english")
        
        tfidf_engine.build_index(processed_titles, processed_keywords, processed_abstracts)
        
        # Get vocabulary size - adjust based on actual TFIDFSearchEngine attributes
        vocab_size = len(tfidf_engine.vocabulario_abstracts) if tfidf_engine.vocabulario_abstracts else 0
        print(f"   ✓ TF-IDF engine ready (vocabulary: {vocab_size} terms)\n")
        
        # Initialize embeddings manager
        print("2. Initializing embeddings manager...")
        try:
            embedding_manager = EmbeddingManager(abstracts)
            
            # Pre-load default model
            default_model = "all-MiniLM-L6-v2"
            print(f"   Pre-loading default model: {default_model}...")
            embedding_manager.preload_model(default_model)
            print()
            
        except Exception as e:
            print(f"   ✗ Error initializing embeddings: {e}")
            import traceback
            traceback.print_exc()
            embedding_manager = None
            print()
        
        print("="*80)
        print("UPSCHOLAR INITIALIZED")
        print("="*80)
        print(f"TF-IDF: {'✓ Available' if tfidf_engine else '✗ Not available'}")
        print(f"Embeddings: {'✓ Available' if embedding_manager else '✗ Not available'}")
        if embedding_manager:
            print(f"Supported models: {', '.join(embedding_manager.get_available_models())}")
            print(f"Loaded models: {', '.join(embedding_manager.get_loaded_models())}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    """Health endpoint"""
    return {
        "message": "UPScholar API v2.0",
        "status": "running",
        "tfidf_ready": tfidf_engine is not None,
        "embedding_ready": embedding_manager is not None,
        "total_documents": len(df_papers) if df_papers is not None else 0,
        "available_models": embedding_manager.get_available_models() if embedding_manager else [],
        "loaded_models": embedding_manager.get_loaded_models() if embedding_manager else []
    }

@app.post("/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """Performs a document search"""
    if df_papers is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        if query.method == "tfidf":
            if tfidf_engine is None:
                raise HTTPException(status_code=500, detail="TF-IDF engine not initialized")
            
            # Process query in THREE PARTS (title, keywords, abstract)
            # In this case, we use the complete query to simulate the three components
            # Ideally the frontend should send these fields separately
            query_title = procesar_texto(query.query, idioma="english")
            query_keywords = procesar_texto(query.query, idioma="english")
            query_abstract = procesar_texto(query.query, idioma="english")
            
            # Search documents with 15-25-60 weighting
            results = tfidf_engine.search(
                query_title, 
                query_keywords, 
                query_abstract, 
                top_k=query.top_k
            )
            top_indices = {idx for idx, _ in results}
            
            # Construir respuesta
            documents = []
            for idx, score in results:
                similar = tfidf_engine.get_similar_documents(
                    idx, top_k=3, exclude_indices=top_indices
                )
                
                similar_docs = [
                    SimilarDocument(
                        id=int(sim_idx),
                        title=df_papers.iloc[sim_idx]['title'],
                        score=float(sim_score)
                    )
                    for sim_idx, sim_score in similar
                ]
                
                doc = Document(
                    id=int(idx),
                    title=df_papers.iloc[idx]['title'],
                    keywords=df_papers.iloc[idx]['keywords'],
                    abstract=df_papers.iloc[idx]['abstract'],
                    score=float(score),
                    similar_docs=similar_docs
                )
                documents.append(doc)
            
            return SearchResponse(
                query=query.query,
                method="tfidf",
                results=documents,
                total_results=len(documents)
            )
            
        elif query.method == "embedding":
            if embedding_manager is None:
                raise HTTPException(
                    status_code=500,
                    detail="Embeddings engine not available"
                )
            
            model_name = query.embedding_model or "all-MiniLM-L6-v2"
            
            print(f"\nSearch with embeddings:")
            print(f"  Query: {query.query}")
            print(f"  Model: {model_name}")
            
            # Search using embeddings
            results = embedding_manager.search(
                query.query,
                model_name=model_name,
                top_k=query.top_k
            )
            
            top_indices = {idx for idx, _ in results}
            
            # Construir respuesta
            documents = []
            for idx, score in results:
                similar = embedding_manager.get_similar_documents(
                    idx,
                    model_name=model_name,
                    top_k=3,
                    exclude_indices=top_indices
                )
                
                similar_docs = [
                    SimilarDocument(
                        id=int(sim_idx),
                        title=df_papers.iloc[sim_idx]['title'],
                        score=float(sim_score)
                    )
                    for sim_idx, sim_score in similar
                ]
                
                doc = Document(
                    id=int(idx),
                    title=df_papers.iloc[idx]['title'],
                    keywords=df_papers.iloc[idx]['keywords'],
                    abstract=df_papers.iloc[idx]['abstract'],
                    score=float(score),
                    similar_docs=similar_docs
                )
                documents.append(doc)
            
            return SearchResponse(
                query=query.query,
                method="embedding",
                results=documents,
                total_results=len(documents),
                embedding_model=model_name
            )
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid method. Use 'tfidf' or 'embedding'"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}")
async def get_document(doc_id: int):
    """Gets a document by its ID"""
    if df_papers is None:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    if doc_id < 0 or doc_id >= len(df_papers):
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc = df_papers.iloc[doc_id]
    return {
        "id": int(doc_id),
        "title": doc['title'],
        "keywords": doc['keywords'],
        "abstract": doc['abstract']
    }

@app.get("/models")
async def get_models():
    """Gets list of available and loaded models"""
    if embedding_manager is None:
        raise HTTPException(status_code=500, detail="Embeddings manager not available")
    
    return {
        "available": embedding_manager.get_available_models(),
        "loaded": embedding_manager.get_loaded_models()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)