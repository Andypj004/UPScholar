# UPScholar

Intelligent search system for scientific articles that combines two approaches:
- Classical search based on TF‑IDF with field weighting (title, keywords, abstract).
- Semantic search using BERT embeddings (SentenceTransformers) with cosine similarity.

Backend in FastAPI, frontend in HTML/CSS/JS, deployment with Docker + Nginx.

---

## Architecture

- Backend (FastAPI API): backend/main.py
  - Health, search, and available models endpoints.
  - Orchestrates TF‑IDF and BERT Embeddings engines.
- TF‑IDF engine: `services.tfidf_search.TFIDFSearchEngine` (backend/services/tfidf_search.py)
  - Indexes over abstracts and weighted similarity 15‑25‑60.
- Embedding components: `services.embedding_manager.EmbeddingManager` (backend/services/embedding_manager.py) and `services.embedding_search.EmbeddingSearchEngine` (backend/services/embedding_search.py)
  - Loads/generates and caches .npy embeddings per model.
- Text preprocessing: `services.text_processing.procesar_texto` and `services.text_processing.procesar_documentos` (backend/services/text_processing.py)
- Frontend: frontend/index.html
  - Method selector (TF‑IDF/Embeddings), embedding model selector, and result rendering.
  - Always uses Nginx proxy path `/api`.

---

## Structure

- backend/main.py: FastAPI API and startup lifecycle.
- Pydantic models: `models.schemas.SearchQuery`, `models.schemas.SearchResponse`, `models.schemas.Document`, `models.schemas.SimilarDocument` (backend/models/schemas.py)
- TF‑IDF: `services.tfidf_search.TFIDFSearchEngine` (backend/services/tfidf_search.py)
- Embeddings:
  - Manager: `services.embedding_manager.EmbeddingManager` (backend/services/embedding_manager.py)
  - Engine: `services.embedding_search.EmbeddingSearchEngine` (backend/services/embedding_search.py)
- Data: data/scientific_papers.csv
- Frontend: frontend/index.html
- Docker: Dockerfile, docker-compose.yml, nginx.conf
- Pre-download models: backend/download_models.py
- Dependencies: backend/requirements.txt

---

## API

- GET `/` — Health
  - Returns status, total documents, and available/loaded models.
- POST `/search` — Search
  - Body: `models.schemas.SearchQuery` (backend/models/schemas.py)
    - query (str), method ("tfidf" | "embedding"), top_k (int), embedding_model (optional).
  - Response: `models.schemas.SearchResponse` (backend/models/schemas.py)
    - Includes list of `models.schemas.Document` and related `models.schemas.SimilarDocument`.
- GET `/documents/{doc_id}` — Document by ID
- GET `/models` — Available and loaded models

Examples:

```bash
# Health
curl -s http://localhost:8000/

# TF-IDF search
curl -s http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"machine learning in healthcare","method":"tfidf","top_k":5}'

# Embeddings search (default model)
curl -s http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"deep learning for medical imaging","method":"embedding","top_k":5,"embedding_model":"all-MiniLM-L6-v2"}'
```

---

## Internal operation

### Preprocessing
- Cleaning, tokenization, stopwords, and stemming: `services.text_processing.procesar_texto`
- Applied per field at startup: `services.text_processing.procesar_documentos`

### TF‑IDF with 15‑25‑60 weighting
- Indexes only over abstracts: `TFIDFSearchEngine.build_index`
- Combined similarity: `TFIDFSearchEngine.calcular_similitud_ponderada`
  - Title: Jaccard (15%)
  - Keywords: Jaccard (25%)
  - Abstract: Cosine over TF‑IDF (60%)
- Weights in `TFIDFSearchEngine.__init__`: `peso_title=0.15`, `peso_keywords=0.25`, `peso_abstract=0.60`
- Formula:
  - S = 0.15 * J(title) + 0.25 * J(keywords) + 0.60 * cos(TF‑IDF abstract)
- Search: `TFIDFSearchEngine.search`
- Similar documents: `TFIDFSearchEngine.get_similar_documents`

### Embedding search (BERT)
- Manager creates/loads engines by model: `EmbeddingManager.get_engine`
- Loads from `.npy` in data/ or generates: `EmbeddingManager._generate_and_save_embeddings`
- Search: `EmbeddingSearchEngine.search`
- Similar documents: `EmbeddingSearchEngine.get_similar_documents`
- Cosine similarity:
  - sim(a, b) = (a · b) / (||a|| * ||b||)
- Supported models: see `EmbeddingManager.available_models`

---

## Frontend

- Single page: frontend/index.html
- Method and model selectors; renders results and related articles.
- JS uses `API_URL = '/api'` (requires Nginx). Locally, use curl/Postman for API; with Docker, access via Nginx.

---

## Getting started

### Installation
1. Clone repo:
   ```bash
   git clone https://github.com/Andypj004/UPScholar.git
    cd UPScholar
    ```
2. Backend dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Run aplication:
   ```bash
   docker-compose up --build
   ```


### Local development (API)
- Health: http://localhost:8000/
- Static UI (without proxy): http://localhost:8000/static/ (note: frontend expects `/api`; for full UI, use Docker/Nginx).

### Docker + Nginx (recommended)
- Dockerfile pre-downloads models, installs dependencies, and serves static frontend.
- Nginx proxy: nginx.conf; services defined in docker-compose.yml.

---

## Data

- Source: data/scientific_papers.csv
- Precomputed embeddings (if present):
  - data/embeddings_all-MiniLM-L6-v2.npy
  - data/embeddings_all-mpnet-base-v2.npy
  - data/embeddings_paraphrase-multilingual-MiniLM-L12-v2.npy
  - data/embeddings_bert-base-nli-mean-tokens.npy

---

## Troubleshooting

- “Embeddings not generated”: the manager will generate and save `.npy`. Check logs in `EmbeddingManager.get_engine`.
- NLTK stopwords locally: download with `nltk.download('stopwords')` and `nltk.download('punkt')`.
- Check status via GET `/`: available/loaded models and total documents.
- First run may be slow due to index/embedding generation; subsequent runs load from cache.s

---

## Extensibility

- Adjust TF‑IDF weights in `TFIDFSearchEngine.__init__`.
- Add models in `EmbeddingManager.available_models`.
- Endpoints and schemas: backend/models/schemas.py.
