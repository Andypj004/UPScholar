import re
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

def limpiar_texto(texto):
    """Removes special characters and numbers, keeps only letters and spaces."""
    texto = re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ\s]', ' ', texto)
    return texto

def limpiezaCaracteresEspeciales(texto, caracteresRemplazar, remplazosCaracteres):
    """Function to perform specific character replacements in a text"""
    try:
        tablaRemplazos = str.maketrans(caracteresRemplazar, remplazosCaracteres)
        textoLimpio = texto.translate(tablaRemplazos)
        textoLimpio = limpiar_texto(textoLimpio)
        textoLimpio = textoLimpio.lower()
        return textoLimpio
    except Exception as error:
        print("Error performing special character cleaning:", error)
        return texto

def procesarStopWords(listaTokens, idioma="spanish", extraStopWords=None):
    """Function to remove stopwords from a token list"""
    try:
        stopWords = stopwords.words(idioma)
        if extraStopWords:
            stopWords.extend(extraStopWords)
        palabrasFiltradas = [token for token in listaTokens if token not in stopWords]
        return palabrasFiltradas
    except Exception as error:
        print("Error processing stopwords:", error)
        return listaTokens

def stemming(listaTokens, idioma="english"):
    """Function to get the stemming of a token list in Spanish or English"""
    try:
        if idioma == "spanish":
            stemmer = SnowballStemmer("spanish")
        elif idioma == "english":
            stemmer = PorterStemmer()
        else:
            raise ValueError("Language not supported for stemming.")
        
        listaStemmings = [stemmer.stem(token) for token in listaTokens]
        return listaStemmings
    
    except Exception as error:
        print("Error getting stemming:", error)
        return listaTokens

def procesar_texto(texto, idioma="english"):
    """Processes a complete text: cleaning, tokenization, stopwords and stemming"""
    texto_limpio = limpiezaCaracteresEspeciales(texto, 'áéíóú-()', 'aeiou   ')
    tokens = texto_limpio.split()
    tokens_sin_stopwords = procesarStopWords(tokens, idioma=idioma)
    tokens_stemmed = stemming(tokens_sin_stopwords, idioma=idioma)
    return tokens_stemmed

def procesar_documentos(textos, idioma="english"):
    """Processes a list of texts"""
    return [procesar_texto(texto, idioma) for texto in textos]