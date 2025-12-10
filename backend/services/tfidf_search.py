import numpy as np
from collections import Counter
from typing import List, Tuple

def obtenerVocabulario(documentos):
    """Función para obtener el vocabulario único a partir de una lista de documentos procesados"""
    try:
        vocabulario = sorted(list({token for doc in documentos for token in doc}))
        return vocabulario
    except Exception as error:
        print("Error en obtener vocabulario:", error)
        return []

def obtenerMatrizFrecuenciaTF(documentos):
    """Función para obtener la matriz de frecuencia TF"""
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
    """Función para obtener la matriz de DF a partir de la matriz de frecuencia TF"""
    try:
        numVocabulario, numDocumentos = matTF.shape
        matDF = np.zeros((numVocabulario, 1), dtype=int)
        for indFila in range(numVocabulario):
            fila = matTF[indFila, :]
            matDF[indFila, 0] = sum(fila != 0)
        return matDF
    except Exception as error:
        print("Error en obtener matriz de DF:", error)
        return None

def obtenerMatrizIDF(matDF, numDocumentos):
    """Función para obtener la matriz de IDF"""
    try:
        matIDF = np.log10(numDocumentos / matDF)
        return matIDF
    except Exception as error:
        print("Error en obtener matriz de IDF:", error)
        return None

def obtenerModeloTFIDF(matWTF, matIDF):
    """Función para obtener la matriz TF-IDF"""
    try:
        matTFIDF = matWTF * matIDF
        return matTFIDF
    except Exception as error:
        print("Error en obtener modelo TF-IDF:", error)
        return None

def obtenerMatrizVUnitario(matTFIDF):
    """Función para obtener la matriz de vectores unitarios"""
    try:
        modulo = np.linalg.norm(matTFIDF, axis=0)
        modulo = np.where(modulo == 0, 1, modulo)
        matVUnitario = matTFIDF / modulo
        return matVUnitario
    except Exception as error:
        print("Error en obtener matriz de vectores unitarios:", error)
        return None

def similitud_jaccard(set1: set, set2: set) -> float:
    """
    Calcula la similitud de Jaccard entre dos conjuntos
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
    """Calcula la matriz de similitud usando coseno vectorial"""
    return np.dot(matriz_v_unitario.T, matriz_v_unitario)

class TFIDFSearchEngine:
    def __init__(self, titles, keywords, abstracts):
        """
        Inicializa el motor de búsqueda TF-IDF con ponderación
        
        Args:
            titles: Lista de títulos de documentos
            keywords: Lista de palabras clave de documentos
            abstracts: Lista de abstracts de documentos
        """
        self.titles = titles
        self.keywords = keywords
        self.abstracts = abstracts
        
        # Guardar versiones procesadas por separado
        self.processed_titles = None
        self.processed_keywords = None
        self.processed_abstracts = None
        
        # Índices TF-IDF solo para abstracts (60%)
        self.vocabulario_abstracts = None
        self.matriz_tfidf_abstracts = None
        self.matriz_v_unitario_abstracts = None
        self.matriz_similitud_abstracts = None
        
        # Pesos para combinación
        self.peso_title = 0.15      # 15%
        self.peso_keywords = 0.25   # 25%
        self.peso_abstract = 0.60   # 60%
        
    def build_index(self, processed_titles, processed_keywords, processed_abstracts):
        """
        Construye los índices necesarios
        
        Args:
            processed_titles: Títulos procesados (tokenizados)
            processed_keywords: Keywords procesadas (tokenizadas)
            processed_abstracts: Abstracts procesados (tokenizados)
        """
        self.processed_titles = processed_titles
        self.processed_keywords = processed_keywords
        self.processed_abstracts = processed_abstracts
        
        print("Construyendo índices TF-IDF...")
        
        # Construir TF-IDF solo para abstracts
        matriz_tf, self.vocabulario_abstracts = obtenerMatrizFrecuenciaTF(processed_abstracts)
        matriz_df = obtenerMatrizDF(matriz_tf)
        matriz_idf = obtenerMatrizIDF(matriz_df, len(processed_abstracts))
        self.matriz_tfidf_abstracts = obtenerModeloTFIDF(matriz_tf, matriz_idf)
        
        # Vectores unitarios para similitud coseno
        self.matriz_v_unitario_abstracts = obtenerMatrizVUnitario(self.matriz_tfidf_abstracts)
        self.matriz_similitud_abstracts = similitud_coseno_vectorial(self.matriz_v_unitario_abstracts)
        
        # Poner diagonal en 0
        np.fill_diagonal(self.matriz_similitud_abstracts, 0)
        
        print(f"✓ Índices construidos (vocabulario: {len(self.vocabulario_abstracts)} términos)")
        
    def calcular_similitud_ponderada(self, query_title: List[str], 
                                     query_keywords: List[str], 
                                     query_abstract: List[str],
                                     doc_index: int) -> float:
        """
        Calcula similitud ponderada combinando:
        - Jaccard para títulos (15%)
        - Jaccard para keywords (25%)
        - Coseno TF-IDF para abstracts (60%)
        
        Args:
            query_title: Tokens del título de la consulta
            query_keywords: Tokens de las keywords de la consulta
            query_abstract: Tokens del abstract de la consulta
            doc_index: Índice del documento a comparar
            
        Returns:
            Similitud ponderada total
        """
        # 1. Similitud Jaccard para título (15%)
        sim_title = similitud_jaccard(
            set(query_title),
            set(self.processed_titles[doc_index])
        )
        
        # 2. Similitud Jaccard para keywords (25%)
        sim_keywords = similitud_jaccard(
            set(query_keywords),
            set(self.processed_keywords[doc_index])
        )
        
        # 3. Similitud Coseno TF-IDF para abstract (60%)
        # Crear vector TF-IDF para la consulta
        indice_vocabulario = {termino: i for i, termino in enumerate(self.vocabulario_abstracts)}
        query_vector = np.zeros(len(self.vocabulario_abstracts))
        
        for termino in query_abstract:
            if termino in indice_vocabulario:
                i = indice_vocabulario[termino]
                query_vector[i] = self.matriz_tfidf_abstracts[i, doc_index]
        
        # Normalizar vector de consulta
        norma_query = np.linalg.norm(query_vector)
        if norma_query > 0:
            query_vector_norm = query_vector / norma_query
        else:
            query_vector_norm = query_vector
        
        # Calcular similitud coseno
        doc_vector_norm = self.matriz_v_unitario_abstracts[:, doc_index]
        sim_abstract = np.dot(query_vector_norm, doc_vector_norm)
        
        # Combinar con pesos
        similitud_total = (
            self.peso_title * sim_title +
            self.peso_keywords * sim_keywords +
            self.peso_abstract * sim_abstract
        )
        
        return similitud_total
    
    def search(self, query_title: List[str], query_keywords: List[str], 
               query_abstract: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Busca documentos similares usando ponderación 15-25-60
        
        Args:
            query_title: Tokens procesados del título
            query_keywords: Tokens procesados de las keywords
            query_abstract: Tokens procesados del abstract
            top_k: Número de documentos a retornar
            
        Returns:
            Lista de tuplas (índice_documento, similitud_ponderada)
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
        
        # Ordenar por similitud descendente
        similitudes.sort(key=lambda x: x[1], reverse=True)
        
        return similitudes[:top_k]
    
    def get_similar_documents(self, doc_index: int, top_k: int = 3, 
                            exclude_indices: set = None) -> List[Tuple[int, float]]:
        """
        Obtiene documentos similares usando similitud de abstracts
        
        Args:
            doc_index: Índice del documento
            top_k: Número de documentos similares a retornar
            exclude_indices: Conjunto de índices a excluir
        
        Returns:
            Lista de tuplas (índice_documento, similitud)
        """
        if exclude_indices is None:
            exclude_indices = set()
        
        # Usar matriz de similitud pre-calculada de abstracts
        similitudes = self.matriz_similitud_abstracts[doc_index, :]
        
        # Crear lista excluyendo índices especificados
        ranking = [
            (i, similitudes[i]) 
            for i in range(len(similitudes)) 
            if i not in exclude_indices and i != doc_index
        ]
        
        # Ordenar por similitud descendente
        ranking.sort(key=lambda x: x[1], reverse=True)
        
        return ranking[:top_k]