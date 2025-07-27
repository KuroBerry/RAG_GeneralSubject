import sys
import os

from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import streamlit as st
import json
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from utils.load_chunks_json import load_chunks_from_json
from utils.bm25 import bm25_tokenize

# Tự động tìm file .env từ thư mục gốc project
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
env_path = os.path.join(root_dir, '.env')
load_dotenv(env_path)

# print(GEMINI_API_KEY)
# print(PINECONE_API_KEY)
# print(HOST_DENSE)
# print(HOST_SPARSE)
def get_gemini_key():
    return os.getenv("GEMINI_API_KEY")

def get_pinecone_key():
    return os.getenv("PINECONE_API_KEY")

def get_host_dense():
    return os.getenv("HOST_DENSE")

def get_host_sparse():
    return os.getenv("HOST_SPARSE")
    
@st.cache_resource
def get_embedding_model():
    """
    Khởi tạo mô hình embedding
    """
    embedding_model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
    return embedding_model

@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key = get_pinecone_key())

    dense_index = pc.Index(host = get_host_dense())
    sparse_index = pc.Index(host = get_host_sparse())
    return dense_index, sparse_index

@st.cache_resource
def get_gemini_model():
    GEMINI_API_KEY = get_gemini_key()
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client

def get_bm25():
    # Lấy raw chunk của các môn đại cương tạo vocabulary
    # Xác định đường dẫn tuyệt đối tới các file JSON
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    lich_su_dang_path = os.path.join(root_dir, 'data', 'LichSuDang', 'Lich_Su_Dang_raw.json')
    triet_hoc_path = os.path.join(root_dir, 'data', 'TrietHoc', 'TrietHoc_raw.json')

    print(lich_su_dang_path, os.path.exists(lich_su_dang_path))
    print(triet_hoc_path, os.path.exists(triet_hoc_path))
    raw_chunk = load_chunks_from_json(lich_su_dang_path) + load_chunks_from_json(triet_hoc_path)

    # Tạo corpus
    corpus_texts = [chunk["content"] for chunk in raw_chunk]
    tokenized_corpus = [bm25_tokenize(text) for text in corpus_texts]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25

def get_vocabulary(bm25):
    """
    Lấy vocabulary từ mô hình BM25
    """
    return list(bm25.idf.keys())