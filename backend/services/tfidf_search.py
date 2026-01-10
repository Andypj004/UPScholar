import numpy as np
from collections import Counter
from typing import List, Tuple

def obtenerVocabulario(documentos):
    """Function to get the unique vocabulary from a list of processed documents"""
    try:
        vocabulario = sorted(list({token for doc in documentos for token in doc}))
        return vocabulario
    except Exception as error:
        print("Error getting vocabulary:", error)
        return []

def obtenerMatrizFrecuenciaTF(documentos):
    """Function to get the TF frequency matrix"""
    vocabulario = obtenerVocabulario(documentos)
    numFilas, numColumnas = len(vocabulario), len(documentos)
    matrizTF = np.zeros((numFilas, numColumnas), dtype=int)
    dicAuxMatTFIndice = {token: i for i, token in enumerate(vocabulario)}
    
    for columnas, doc in enumerate(documentos):
        conteo = Counter(doc)
        for token in conteo:
            matrizTF[dicAuxMatTFIndice[token], columnas] = conteo[token]
    
    return matrizTF, vocabulario

def obtenerMatrizDF(matTF):
    """Function to get the DF matrix from the TF frequency matrix"""
    try:
        numVocabulario, numDocumentos = matTF.shape
        matDF = np.zeros((numVocabulario, 1), dtype=int)
        for indFila in range(numVocabulario):
            fila = matTF[indFila, :]
            matDF[indFila, 0] = sum(fila != 0)
        return matDF
    except Exception as error:
        print("Error getting DF matrix:", error)
        return None

def obtenerMatrizIDF(matDF, numDocumentos):
    """Function to get the IDF matrix"""
    try:
        matIDF = np.log10(numDocumentos / matDF)
        return matIDF
    except Exception as error:
        print("Error getting IDF matrix:", error)
        return None

def obtenerModeloTFIDF(matWTF, matIDF):
    """Function to get the TF-IDF matrix"""
    try:
        matTFIDF = matWTF * matIDF
        return matTFIDF
    except Exception as error:
        print("Error getting TF-IDF model:", error)
        return None

def obtenerMatrizVUnitario(matTFIDF):
    """Function to get the unit vectors matrix"""
    try:
        modulo = np.linalg.norm(matTFIDF, axis=0)
        modulo = np.where(modulo == 0, 1, modulo)
        matVUnitario = matTFIDF / modulo
        return matVUnitario
    except Exception as error:
        print("Error getting unit vectors matrix:", error)
        return None

def similitud_jaccard(set1: set, set2: set) -> float:
    """
    Calculates Jaccard similarity between two sets
    Jaccard = |A ∩ B| / |A ∪ B|
    """
    if len(set1) == 0 and len(set2) == 0:
        return 0.0
    
    interseccion = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return interseccion / union

def similitud_coseno_vectorial(matriz_v_unitario):
    """Calculates similarity matrix using vectorial cosine"""
    return np.dot(matriz_v_unitario.T, matriz_v_unitario)

class TFIDFSearchEngine:
    def __init__(self, titles, keywords, abstracts):
        """
        Initializes the TF-IDF search engine with weighting
        
        Args:
            titles: List of document titles
            keywords: List of document keywords
            abstracts: List of document abstracts
        """
        self.titles = titles
        self.keywords = keywords
        self.abstracts = abstracts
        
        # Save separately processed versions
        self.processed_titles = None
        self.processed_keywords = None
        self.processed_abstracts = None
        
        # TF-IDF indices only for abstracts (60%)
        self.vocabulario_abstracts = None
        self.matriz_tfidf_abstracts = None
        self.matriz_v_unitario_abstracts = None
        self.matriz_similitud_abstracts = None
        # IDF matrix for abstracts
        self.matriz_idf_abstracts = None
        self.indice_vocabulario_abstracts = None
        # Weights for combination
        self.peso_title = 0.15      # 15%
        self.peso_keywords = 0.25   # 25%
        self.peso_abstract = 0.60   # 60%
        
    def build_index(self, processed_titles, processed_keywords, processed_abstracts):
        """
        Builds the necessary indices
        
        Args:
            processed_titles: Processed titles (tokenized)
            processed_keywords: Processed keywords (tokenized)
            processed_abstracts: Processed abstracts (tokenized)
        """
        self.processed_titles = processed_titles
        self.processed_keywords = processed_keywords
        self.processed_abstracts = processed_abstracts
        
        print("Building TF-IDF indices...")
        
        # Build TF-IDF only for abstracts
        matriz_tf, self.vocabulario_abstracts = obtenerMatrizFrecuenciaTF(processed_abstracts)
        matriz_df = obtenerMatrizDF(matriz_tf)
        matriz_idf = obtenerMatrizIDF(matriz_df, len(processed_abstracts))
        self.matriz_idf_abstracts = matriz_idf  # guardar para consultas
        self.matriz_tfidf_abstracts = obtenerModeloTFIDF(matriz_tf, matriz_idf)
        # Map vocabulary to indices
        self.indice_vocabulario_abstracts = {t: i for i, t in enumerate(self.vocabulario_abstracts)}
        
        # Unit vectors for cosine similarity
        self.matriz_v_unitario_abstracts = obtenerMatrizVUnitario(self.matriz_tfidf_abstracts)
        self.matriz_similitud_abstracts = similitud_coseno_vectorial(self.matriz_v_unitario_abstracts)
        
        # Set diagonal to 0
        np.fill_diagonal(self.matriz_similitud_abstracts, 0)
        
        print(f"✓ Indices built (vocabulary: {len(self.vocabulario_abstracts)} terms)")
        
    def calcular_similitud_ponderada(self, query_title: List[str], 
                                     query_keywords: List[str], 
                                     query_abstract: List[str],
                                     doc_index: int) -> float:
        """
        Calculates weighted similarity combining:
        - Jaccard for titles (15%)
        - Jaccard for keywords (25%)
        - Cosine TF-IDF for abstracts (60%)
        
        Args:
            query_title: Tokens of query title
            query_keywords: Tokens of query keywords
            query_abstract: Tokens of query abstract
            doc_index: Document index to compare
            
        Returns:
            Total weighted similarity
        """
        # 1. Jaccard similarity for title (15%)
        sim_title = similitud_jaccard(
            set(query_title),
            set(self.processed_titles[doc_index])
        )
        
        # 2. Jaccard similarity for keywords (25%)
        sim_keywords = similitud_jaccard(
            set(query_keywords),
            set(self.processed_keywords[doc_index])
        )
        
        # 3. Cosine similarity for abstracts (60%)
        V = len(self.vocabulario_abstracts)
        query_tf = np.zeros(V, dtype=float)
        for termino in query_abstract:
            idx = self.indice_vocabulario_abstracts.get(termino)
            if idx is not None:
                query_tf[idx] += 1.0

        # TF-IDF for query
        # self.matriz_idf_abstracts tiene forma (V,1)
        query_tfidf = query_tf * self.matriz_idf_abstracts.flatten()

        # Normalize query vector
        norma_query = np.linalg.norm(query_tfidf)
        if norma_query > 0:
            query_vector_norm = query_tfidf / norma_query
        else:
            query_vector_norm = query_tfidf

        doc_vector_norm = self.matriz_v_unitario_abstracts[:, doc_index]
        sim_abstract = float(np.dot(query_vector_norm, doc_vector_norm))

        # Weighted total similarity
        similitud_total = (
            self.peso_title * sim_title +
            self.peso_keywords * sim_keywords +
            self.peso_abstract * sim_abstract
        )
        return similitud_total
    
    def search(self, query_title: List[str], query_keywords: List[str], 
               query_abstract: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Searches similar documents using 15-25-60 weighting
        
        Args:
            query_title: Processed tokens of title
            query_keywords: Processed tokens of keywords
            query_abstract: Processed tokens of abstract
            top_k: Number of documents to return
            
        Returns:
            List of tuples (document_index, weighted_similarity)
        """
        similitudes = []
        
        for doc_idx in range(len(self.processed_abstracts)):
            sim = self.calcular_similitud_ponderada(
                query_title,
                query_keywords,
                query_abstract,
                doc_idx
            )
            similitudes.append((doc_idx, sim))
        
        # Sort by descending similarity
        similitudes.sort(key=lambda x: x[1], reverse=True)
        
        return similitudes[:top_k]
    
    def get_similar_documents(self, doc_index: int, top_k: int = 3, 
                            exclude_indices: set = None) -> List[Tuple[int, float]]:
        """
        Gets similar documents using abstract similarity
        
        Args:
            doc_index: Document index
            top_k: Number of similar documents to return
            exclude_indices: Set of indices to exclude
        
        Returns:
            List of tuples (document_index, similarity)
        """
        if exclude_indices is None:
            exclude_indices = set()
        
        # Use pre-calculated similarity matrix of abstracts
        similitudes = self.matriz_similitud_abstracts[doc_index, :]
        
        # Create list excluding specified indices
        ranking = [
            (i, similitudes[i]) 
            for i in range(len(similitudes)) 
            if i not in exclude_indices and i != doc_index
        ]
        
        # Sort by descending similarity
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking[:top_k]