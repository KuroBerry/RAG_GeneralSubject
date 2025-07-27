#Tạo đương dẫn chung để đọc utils
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Load các thư viện cần thiết
import json
from pinecone.grpc import PineconeGRPC as Pinecone
import os
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai

# Đọc các file JSON
#Tạo hàm đọc
from utils.bm25 import text_to_sparse_vector_bm25
from agent import get_router
import retriever.cache_data as cache_data       

def semantic_dense(input_query, namespace, k, embedding_model = None):
    dense_index = cache_data.get_pinecone_index()[0]
    dense_results = dense_index.query(
        namespace= namespace,
        vector=embedding_model.encode(input_query, convert_to_tensor=False).tolist(),
        top_k=k,
        include_metadata=True,
        # fields=["category", "chunk_text"]
    )
    return dense_results



def lexical_sparse(input_querry, namespace, bm25, vocabulary, k):
    sparse_index = cache_data.get_pinecone_index()[1]
    sparse_vector = text_to_sparse_vector_bm25(input_querry, bm25, vocabulary)

    sparse_results = sparse_index.query(
        namespace=namespace,
        sparse_vector=sparse_vector,
        top_k=k,
        include_metadata=True,
    )

    return sparse_results

def merge_chunks(h1, h2):
    """Get the unique hits from two search results and return them as single array of {'_id', 'chunk_text'} dicts, printing each dict on a new line."""
    # Deduplicate by _id
    deduped_hits = {hit['id']: hit for hit in h1['matches'] + h2['matches']}.values()
    print
    # Sort by _score descending
    sorted_hits = sorted(deduped_hits, key=lambda x: x['score'], reverse=True)
    # Transform to format for reranking
    result = [{'id': hit['id'], 'content': hit['metadata']['content']} for hit in sorted_hits]
    return result

def hybrid_retriever(input_query, model_choice, top_k):

    #Lấy ra các biến cần thiết
    embedding_model = cache_data.get_embedding_model()
    bm25 = cache_data.get_bm25()
    vocabulary = cache_data.get_vocabulary(bm25)

    #Lấy namespace để truyền vào tìm kiếm vector
    namespace = get_router(input_query, model_choice)

    # Thực hiện tìm kiếm Semantic và Lexical
    dense_results = semantic_dense(input_query, namespace, top_k, embedding_model)
    sparse_results = lexical_sparse(input_query, namespace, bm25, vocabulary, top_k)

    # Kết hợp kết quả từ hai phương pháp
    results = merge_chunks(dense_results, sparse_results)

    return results

# #Gọi mô hình Embedding
# input_querry = "ĐCSVN thành lập năm nào?"
# results = hybrid_retriever(input_querry, 'gemini-2.5-flash', 10)

# print('[\n   ' + ',\n   '.join(str(obj) for obj in results) + '\n]')
# print(len (results))