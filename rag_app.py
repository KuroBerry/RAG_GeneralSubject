import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="RAG Political Science Chatbot",
    page_icon="🎓",
    layout="centered"
)

import sys
import os
import time
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Demo functions first (in case real imports fail)
def demo_parent_document_search(query, **kwargs):
    """Demo version of parent_document_search"""
    time.sleep(1)  # Simulate processing
    
    demo_responses = {
        "đảng cộng sản": "Đảng Cộng sản Việt Nam được thành lập ngày 3/2/1930 tại Hồng Kông do Chủ tịch Hồ Chí Minh sáng lập.",
        "tư tưởng hồ chí minh": "Tư tưởng Hồ Chí Minh là hệ thống quan điểm toàn diện về cách mạng Việt Nam, bao gồm độc lập dân tộc và chủ nghĩa xã hội.",
        "triết học": "Triết học Marxism-Leninism là nền tảng tư tưởng của Đảng, giúp nhận thức đúng đắn về thế giới và xã hội."
    }
    
    query_lower = query.lower()
    content = "Demo content cho câu hỏi: " + query
    for keyword, response in demo_responses.items():
        if keyword in query_lower:
            content = response
            break
    
    return [{
        'parent_chunk': {'metadata': {'content': content, 'chapter': 'Demo Chapter', 'section': 'Demo Section'}},
        'parent_id': f'demo_{abs(hash(query)) % 1000:03d}',
        'score': 0.95,
        'child_count': 3
    }]

def demo_generate_answer(query, context):
    """Demo version of generate_answer"""
    time.sleep(1)  # Simulate processing
    
    context_text = ""
    for metadata, parent_id in context:
        content = metadata.get('content', '')
        context_text += f"📖 {content[:150]}... (ID: {parent_id})\n\n"
    
    return f"""**Câu hỏi:** {query}

**Thông tin tìm được:**
{context_text}

**Trả lời:** Dựa trên tài liệu tìm được, đây là câu trả lời demo cho câu hỏi của bạn.

⚠️ **Demo Mode**: Để có câu trả lời chính xác, vui lòng cấu hình API keys trong file .env

*Nguồn: RAG Demo System*"""

# Try to import real functions
RETRIEVAL_AVAILABLE = False
retrieval_status = "⚠️ Initializing..."

# Defer imports that might contain Streamlit decorators
parent_document_search = None
generate_answer = None
initialize_connections = None

try:
    # Load environment variables first
    from dotenv import load_dotenv
    import os
    
    # Try multiple paths for .env
    env_loaded = False
    for env_path in ['.env', '../.env', './.env']:
        try:
            load_dotenv(env_path)
            if os.getenv('PINECONE_API_KEY') and os.getenv('HOST_DENSE') and os.getenv('GEMINI_API_KEY'):
                env_loaded = True
                break
        except:
            continue
    
    if not env_loaded:
        raise Exception("Environment variables not found")
    
    # Import retrieval functions AFTER page config
    from retriever.parent_retrieval import parent_document_search, generate_answer, initialize_connections
    
    # Try to initialize
    initialize_connections()
    RETRIEVAL_AVAILABLE = True
    retrieval_status = "✅ RAG system active with real APIs"
    
except Exception as e:
    retrieval_status = f"⚠️ Demo mode: {str(e)[:60]}..."

# CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 10%;
    }
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 10%;
    }
    .source-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #333333;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .source-card strong {
        color: #007bff;
        font-weight: 600;
    }
    
    /* Improve readability for expandable sections */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        border-radius: 5px !important;
    }
    
    /* Custom styling for metrics */
    .metric-container {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# Header
st.title("🎓 RAG Political Science Chatbot")

# Debug info
with st.expander("🔧 System Debug Info", expanded=False):
    st.code(f"Status: {retrieval_status}")
    
    # Check .env file
    import os
    env_file_exists = os.path.exists('.env')
    st.write(f"📁 .env file exists: {'✅' if env_file_exists else '❌'}")
    
    if env_file_exists:
        from dotenv import load_dotenv
        load_dotenv()
        st.write(f"🔑 PINECONE_API_KEY: {'✅ Found' if os.getenv('PINECONE_API_KEY') else '❌ Missing'}")
        st.write(f"🌐 HOST_DENSE: {'✅ Found' if os.getenv('HOST_DENSE') else '❌ Missing'}")
        st.write(f"🤖 GEMINI_API_KEY: {'✅ Found' if os.getenv('GEMINI_API_KEY') else '❌ Missing'}")
    
    # Cache status
    st.subheader("🚀 Cache Status")
    try:
        # Try to get cache stats if available
        cache_info = getattr(st.cache_data, 'get_stats', lambda: {})()
        st.write(f"📊 Cache entries: {len(cache_info) if cache_info else 'N/A'}")
    except:
        st.write("📊 Cache monitoring: Available")
    
    if RETRIEVAL_AVAILABLE:
        st.write("✅ Embedding model: Cached")
        st.write("✅ Pinecone connection: Cached") 
        st.write("✅ Gemini client: Cached")
    else:
        st.write("⚠️ Cache status: Demo mode")

st.markdown(f"**Status:** {retrieval_status}")

# Mode indicator
if RETRIEVAL_AVAILABLE:
    st.success("🚀 **Full RAG Mode**: Parent Document Retrieval Active")
else:
    st.warning("🔧 **Demo Mode**: Showing sample responses")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🔧 Settings")
    
    # RAG parameters
    top_k = st.slider("Results", 1, 10, 5)
    alpha = st.slider("Score weight", 0.0, 1.0, 0.7, 0.1)
    
    # Namespaces
    st.subheader("📚 Data Source")
    child_ns = st.selectbox("Child namespace", 
                           ["lich-su-dang-children", "triet-hoc-children", "tu-tuong-hcm-children"])
    parent_ns = st.selectbox("Parent namespace", 
                            ["lich-su-dang", "triet-hoc", "tu-tuong-hcm"])
    
    # Stats
    st.subheader("📊 Statistics")
    st.metric("Total Queries", st.session_state.total_queries)
    
    if st.session_state.messages:
        # Calculate average response time
        response_times = []
        for msg in st.session_state.messages:
            if msg["role"] == "assistant" and "response_time" in msg:
                try:
                    rt = float(msg["response_time"])
                    response_times.append(rt)
                except:
                    pass
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
            st.metric("Last Response", f"{response_times[-1]:.2f}s")
    
    # Cache efficiency
    if RETRIEVAL_AVAILABLE:
        st.success("🚀 Using cached resources")
    else:
        st.info("🔧 Demo mode active")
    
    # Clear button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()
    
    # Clear cache button (for debugging)
    if st.button("🧹 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
        st.rerun()

# Display messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑‍🎓 You:</strong><br>{message["content"]}
            <small style="opacity:0.8;display:block;margin-top:0.5rem;">
                {message.get("timestamp", "")}
            </small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Assistant:</strong><br>{message["content"]}
            <small style="opacity:0.8;display:block;margin-top:0.5rem;">
                ⏱️ {message.get("response_time", "N/A")}s | 📚 {message.get("source_count", 0)} sources
            </small>
        </div>
        """, unsafe_allow_html=True)
        
        # Sources
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 {len(message['sources'])} Sources"):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <div style="margin-bottom: 0.5rem;">
                            <strong>Source {i}</strong> | 
                            <span style="color: #28a745;">Score: {src['score']:.4f}</span> | 
                            <span style="color: #6c757d;">Children: {src['child_count']}</span>
                        </div>
                        <div style="margin-bottom: 0.3rem;">
                            <strong>ID:</strong> <code style="background: #e9ecef; padding: 2px 4px; border-radius: 3px; color: #495057;">{src['parent_id']}</code>
                        </div>
                        <div>
                            <strong>Content:</strong><br>
                            <span style="color: #495057; line-height: 1.4;">{src['content'][:300]}...</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Input
st.markdown("### 💬 Ask a Question")
user_input = st.text_input("", placeholder="e.g., Đảng Cộng sản Việt Nam được thành lập khi nào?")

col1, col2 = st.columns([3, 1])
with col1:
    send_btn = st.button("🚀 Send", use_container_width=True, type="primary")
with col2:
    help_btn = st.button("💡 Examples", use_container_width=True)

# Example questions
if help_btn:
    examples = [
        "Đảng Cộng sản Việt Nam được thành lập khi nào?",
        "Tư tưởng Hồ Chí Minh là gì?",
        "Triết học Marxism-Leninism có đặc điểm gì?",
        "Vai trò của Đảng trong cách mạng Việt Nam?"
    ]
    
    st.markdown("**📝 Example Questions:**")
    for ex in examples:
        if st.button(f"• {ex}", key=f"ex_{abs(hash(ex))%1000}"):
            # Add to messages
            st.session_state.messages.append({
                "role": "user", 
                "content": ex,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            # Process immediately
            with st.spinner("🔍 Processing..."):
                start_time = time.time()
                
                if RETRIEVAL_AVAILABLE:
                    results = parent_document_search(ex, child_ns, parent_ns, top_k, alpha)
                else:
                    results = demo_parent_document_search(ex)
                
                if results:
                    context = [(r['parent_chunk']['metadata'], r['parent_id']) for r in results]
                    if RETRIEVAL_AVAILABLE:
                        answer = generate_answer(ex, context)
                    else:
                        answer = demo_generate_answer(ex, context)
                    
                    response_time = time.time() - start_time
                    st.session_state.total_queries += 1
                    
                    sources = [{
                        'parent_id': r['parent_id'],
                        'score': r['score'],
                        'child_count': r['child_count'],
                        'content': r['parent_chunk']['metadata'].get('content', 'No content')
                    } for r in results]
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "response_time": f"{response_time:.2f}",
                        "source_count": len(results),
                        "sources": sources
                    })
            st.rerun()

# Process input
if send_btn and user_input.strip():
    # Add user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user", 
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Process
    with st.spinner("🔍 Searching documents..."):
        start_time = time.time()
        
        try:
            # Search
            if RETRIEVAL_AVAILABLE:
                results = parent_document_search(user_input, child_ns, parent_ns, top_k, alpha)
            else:
                results = demo_parent_document_search(user_input)
            
            if results:
                # Generate answer
                context = [(r['parent_chunk']['metadata'], r['parent_id']) for r in results]
                
                if RETRIEVAL_AVAILABLE:
                    answer = generate_answer(user_input, context)
                else:
                    answer = demo_generate_answer(user_input, context)
                
                response_time = time.time() - start_time
                st.session_state.total_queries += 1
                
                # Prepare sources
                sources = [{
                    'parent_id': r['parent_id'],
                    'score': r['score'],
                    'child_count': r['child_count'],
                    'content': r['parent_chunk']['metadata'].get('content', 'No content')
                } for r in results]
                
                # Add response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "response_time": f"{response_time:.2f}",
                    "source_count": len(results),
                    "sources": sources
                })
            else:
                # No results
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Xin lỗi, không tìm thấy thông tin liên quan. Vui lòng thử câu hỏi khác.",
                    "response_time": "N/A",
                    "source_count": 0,
                    "sources": []
                })
                
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Lỗi xử lý: {str(e)}",
                "response_time": "N/A",
                "source_count": 0,
                "sources": []
            })
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>🎓 RAG Political Science Assistant | Parent Document Retrieval</small>
</div>
""", unsafe_allow_html=True)
