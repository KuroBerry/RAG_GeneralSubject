#Tạo đương dẫn chung để đọc utils
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

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
from utils.load_chunks_json import load_chunks_from_json
from utils.save_chunks_json import save_chunks_to_json
from utils.bm25 import bm25_tokenize, text_to_sparse_vector_bm25
import cache_data


def semantic_dense(input_query, namespace, embedding_model = None):
    dense_index = cache_data.get_pinecone_index()[0]
    dense_results = dense_index.query(
        namespace= namespace,
        vector=embedding_model.encode(input_query, convert_to_tensor=False).tolist(),
        top_k=10,
        include_metadata=True,
        # fields=["category", "chunk_text"]
    )
    return dense_results

# Lấy raw chunk của các môn đại cương tạo vocabulary
raw_chunk = load_chunks_from_json(r"../data/LichSuDang/Lich_Su_Dang_raw.json") + load_chunks_from_json(r"../data/TrietHoc/TrietHoc_raw.json")

# Tạo corpus
corpus_texts = [chunk["content"] for chunk in raw_chunk]
tokenized_corpus = [bm25_tokenize(text) for text in corpus_texts]

bm25 = BM25Okapi(tokenized_corpus)
vocabulary = list(bm25.idf.keys())

def lexical_sparse(input_querry, namespace, bm25, vocabulary):
    sparse_index = cache_data.get_pinecone_index()[1]
    sparse_vector = text_to_sparse_vector_bm25(input_querry, bm25, vocabulary)

    sparse_results = sparse_index.query(
        namespace=namespace,
        sparse_vector=sparse_vector,
        top_k=10,
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

def hybrid_retriever(input_query, embedding_model, bm25, vocabulary):
    #Lấy namespace để truyền vào tìm kiếm vector
    namespace = subject_classification(input_query)

    # Thực hiện tìm kiếm Semantic và Lexical
    dense_results = semantic_dense(input_query, namespace, embedding_model)
    sparse_results = lexical_sparse(input_query, namespace, bm25, vocabulary)

    # Kết hợp kết quả từ hai phương pháp
    results = merge_chunks(dense_results, sparse_results)

    # Lưu kết quả vào file JSON
    save_chunks_to_json(results, "../context/hybrid_search_results.json")

    return results

# input_querry = "Quân Tưởng"
# results = hybrid_retriever(input_querry, embedding_model, bm25, vocabulary)

# print('[\n   ' + ',\n   '.join(str(obj) for obj in results) + '\n]')
# print(len (results))