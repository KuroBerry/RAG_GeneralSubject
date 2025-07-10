import numpy as np
from rank_bm25 import BM25Okapi

def bm25_tokenize(text):
    return text.lower().split()

def text_to_sparse_vector_bm25(text, bm25, vocabulary):
    tokens = bm25_tokenize(text)
    vector = np.zeros(len(vocabulary))
    for i, word in enumerate(vocabulary):
        idf = bm25.idf.get(word, 0)
        tf = tokens.count(word)
        vector[i] = idf * tf
    indices = vector.nonzero()[0].tolist()
    values = vector[indices].tolist()
    return {"indices": indices, "values": values}