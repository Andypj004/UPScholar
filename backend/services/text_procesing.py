import re
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

def limpiar_texto(texto):
    """Elimina caracteres especiales y números, mantiene solo letras y espacios."""
    texto = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', ' ', texto)
    return texto

def limpiezaCaracteresEspeciales(texto, caracteresRemplazar, remplazosCaracteres):
    """Función para realizar reemplazos específicos de caracteres en un texto"""
    try:
        tablaRemplazos = str.maketrans(caracteresRemplazar, remplazosCaracteres)
        textoLimpio = texto.translate(tablaRemplazos)
        textoLimpio = limpiar_texto(textoLimpio)
        textoLimpio = textoLimpio.lower()
        return textoLimpio
    except Exception as error:
        print("Error al realizar limpieza de caracteres especiales:", error)
        return texto

def procesarStopWords(listaTokens, idioma="spanish", extraStopWords=None):
    """Función para eliminar stopwords de una lista de tokens"""
    try:
        stopWords = stopwords.words(idioma)
        if extraStopWords:
            stopWords.extend(extraStopWords)
        palabrasFiltradas = [token for token in listaTokens if token not in stopWords]
        return palabrasFiltradas
    except Exception as error:
        print("Error al procesar stopwords:", error)
        return listaTokens

def stemming(listaTokens, idioma="english"):
    """Función para obtener el stemming de una lista de tokens en español o inglés"""
    try:
        if idioma == "spanish":
            stemmer = SnowballStemmer("spanish")
        elif idioma == "english":
            stemmer = PorterStemmer()
        else:
            raise ValueError("Idioma no soportado para stemming.")
        
        listaStemmings = [stemmer.stem(token) for token in listaTokens]
        return listaStemmings
    
    except Exception as error:
        print("Error al obtener el stemming:", error)
        return listaTokens

def procesar_texto(texto, idioma="english"):
    """Procesa un texto completo: limpieza, tokenización, stopwords y stemming"""
    texto_limpio = limpiezaCaracteresEspeciales(texto, 'áéíóú-()', 'aeiou   ')
    tokens = texto_limpio.split()
    tokens_sin_stopwords = procesarStopWords(tokens, idioma=idioma)
    tokens_stemmed = stemming(tokens_sin_stopwords, idioma=idioma)
    return tokens_stemmed

def procesar_documentos(textos, idioma="english"):
    """Procesa una lista de textos"""
    return [procesar_texto(texto, idioma) for texto in textos]