import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from pathlib import Path
import streamlit as st

# Load environment variables
import os
from pathlib import Path

# Find .env file in multiple locations
env_paths = [
    Path(__file__).parent.parent / '.env',  # Root directory
    Path(__file__).parent / '.env',         # retriever directory
    Path('.env'),                           # Current directory
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()  # Last resort - search automatically

# Get environment variables and clean them
pinecone_api_key = os.getenv("PINECONE_API_KEY", "").strip().strip('"\'')
host_dense = os.getenv("HOST_DENSE", "").strip().strip('"\'')
gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"\'')

print(f"Debug - Loaded keys: PINECONE={'✓' if pinecone_api_key else '✗'}, HOST={'✓' if host_dense else '✗'}, GEMINI={'✓' if gemini_api_key else '✗'}")

# Check if environment variables are loaded
if not pinecone_api_key or not host_dense or not gemini_api_key:
    print("Warning: Missing environment variables. Please check your .env file.")
    print("Required variables: PINECONE_API_KEY, HOST_DENSE, GEMINI_API_KEY")
    # For development, you can set dummy values or skip initialization

# Initialize connections only if keys are available
pc = None
dense_index = None
embedding_model = None
gemini_client = None

@st.cache_resource
def get_pinecone_connection():
    """Cache Pinecone connection to avoid re-initialization"""
    if not all([pinecone_api_key, host_dense]):
        raise ValueError("Missing Pinecone environment variables")
    
    pc = Pinecone(api_key=pinecone_api_key)
    dense_index = pc.Index(host=host_dense)
    print("✅ Pinecone connection initialized and cached")
    return pc, dense_index

@st.cache_resource
def get_embedding_model():
    """Cache embedding model to avoid re-loading"""
    print("🔄 Loading embedding model...")
    model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
    model.max_seq_length = 2048
    print("✅ Embedding model loaded and cached")
    return model

@st.cache_resource
def get_gemini_client():
    """Cache Gemini client"""
    if not gemini_api_key:
        raise ValueError("Missing Gemini API key")
    
    client = genai.Client(api_key=gemini_api_key)
    print("✅ Gemini client initialized and cached")
    return client

def initialize_connections():
    """Initialize all connections using cached resources"""
    global pc, dense_index, embedding_model, gemini_client
    
    if not all([pinecone_api_key, host_dense, gemini_api_key]):
        raise ValueError("Missing required environment variables. Please check your .env file.")
    
    # Get cached resources
    pc, dense_index = get_pinecone_connection()
    embedding_model = get_embedding_model()
    gemini_client = get_gemini_client()
    
    print("🚀 All connections initialized using cache")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def embed_query(query):
    """Cache query embeddings to avoid re-encoding same questions"""
    embedding_model = get_embedding_model()
    return embedding_model.encode(query).tolist()

@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_parent_chunks(parent_ids, parent_namespace):
    """Cache parent chunks fetching to avoid repeated DB calls"""
    try:
        _, dense_index = get_pinecone_connection()
        
        # Batch fetch parent chunks
        chunks = {}
        try:
            parent_fetch_result = dense_index.fetch(ids=parent_ids, namespace=parent_namespace)
            chunks = parent_fetch_result.vectors
        except Exception as e:
            print(f"Error fetching parent chunks: {e}")
        
        return chunks
    except Exception as e:
        print(f"Error in fetch_parent_chunks: {e}")
        return {}

def parent_document_search(query, child_namespace="lich-su-dang-children", parent_namespace="lich-su-dang", top_k=10, alpha=0.7):
    """
    Search parent documents using child chunks with cached resources
    """
    global dense_index, embedding_model
    
    # Get cached resources directly instead of using global variables
    try:
        _, dense_index = get_pinecone_connection()
        # Use cached embedding function
        query_vector = embed_query(query)
    except Exception as e:
        print(f"Error accessing cached resources: {e}")
        # Fallback to global initialization
        if dense_index is None or embedding_model is None:
            initialize_connections()
        query_vector = embedding_model.encode(query).tolist()
    
    # Search child chunks
    child_results = dense_index.query(
        vector=query_vector,
        top_k=top_k*2,  # Get more child chunks to find diverse parents
        include_metadata=True,
        namespace=child_namespace
    )
    
    if not child_results.matches:
        return []
    
    # Aggregate scores by parent_id
    parent_scores = {}
    for match in child_results.matches:
        parent_id = match.metadata.get('parent_id')
        if parent_id:
            if parent_id not in parent_scores:
                parent_scores[parent_id] = {
                    'total_score': 0,
                    'child_count': 0,
                    'best_score': 0
                }
            
            parent_scores[parent_id]['total_score'] += match.score
            parent_scores[parent_id]['child_count'] += 1
            parent_scores[parent_id]['best_score'] = max(parent_scores[parent_id]['best_score'], match.score)
    
    if not parent_scores:
        return []
    
    # Fetch parent chunks from Pinecone
    parent_ids = list(parent_scores.keys())
    parent_chunks = {}
    
    # Fetch in batches
    for i in range(0, len(parent_ids), 100):
        batch_ids = parent_ids[i:i+100]
        result = fetch_parent_chunks(batch_ids, parent_namespace)
        parent_chunks.update(result)
    
    # Calculate final scores and rank
    ranked_parents = []
    for parent_id in parent_ids:
        if parent_id in parent_chunks:
            scores = parent_scores[parent_id]
            avg_score = scores['total_score'] / scores['child_count']
            normalized_child_count = min(scores['child_count'] / 10.0, 1.0)
            
            # Combine average score with child count
            final_score = alpha * avg_score + (1 - alpha) * normalized_child_count
            
            ranked_parents.append({
                'parent_chunk': parent_chunks[parent_id],
                'parent_id': parent_id,
                'score': final_score,
                'avg_child_score': avg_score,
                'best_child_score': scores['best_score'],
                'total_child_score': scores['total_score'],
                'child_count': scores['child_count']
            })
    
    # Sort by score and return top_k
    ranked_parents.sort(key=lambda x: x['score'], reverse=True)
    return ranked_parents[:top_k]

def generate_answer(query, context):
    """
    Generate answer using Gemini with cached client
    """
    global gemini_client
    
    # Get cached Gemini client directly
    try:
        gemini_client = get_gemini_client()
    except Exception as e:
        print(f"Error accessing cached Gemini client: {e}")
        # Fallback to global initialization
        if gemini_client is None:
            initialize_connections()
    
    prompt = f"""
Mày là một chuyên gia trong việc trả lời các môn học đại cương về chính trị bậc đại học không chính quy.

Dưới đây chính là query của người dùng về các câu hỏi liên quan tới các môn chính trị:
{query}

Còn dưới đây là những context được cung cấp để mày trả lời câu hỏi của người dùng:
{context}

Yêu cầu khi trả lời:
- Tóm tắt và tổng hợp thông tin từ các đoạn context có liên quan.
- Hạn chế sao chép nguyên văn toàn bộ một đoạn nào từ context.
- Văn phong rõ ràng, súc tích, mang tính học thuật.
- Hãy ghi nguồn gốc của thông tin trong câu trả lời bằng id của đoạn văn bản trong context, đặt id ở cuối thông tin đó.
- Dựa vào những thông tin trên, mày hãy thực hiện trả lời câu hỏi của người dùng. Chỉ trả lời theo nội dung context cung cấp. Nếu những nội dung đó không liên quan đến câu hỏi thì trả lời "Context không liên quan".
"""
    
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    
    return response.text

# Test the functions
if __name__ == "__main__":
    query = "Đảng Cộng sản Việt Nam được thành lập khi nào?"
    
    # Search parent documents
    results = parent_document_search(query, top_k=5)
    
    print(f"Query: {query}")
    print(f"Found {len(results)} parent chunks")
    
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Parent ID: {result['parent_id']}")
        print(f"Child count: {result['child_count']}")
        metadata = result['parent_chunk']['metadata']
        content = metadata.get('content', '')[:200] + "..."
        print(f"Content: {content}")
    
    # Generate answer
    if results:
        context = [(result['parent_chunk']['metadata'], result['parent_id']) for result in results]
        answer = generate_answer(query, context)
        print(f"\nGenerated Answer:")
        print(answer)
