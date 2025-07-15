"""
Simple test script for RAG app
"""
import os
import sys
from pathlib import Path

print("🔧 Testing RAG Environment Setup...")
print("=" * 50)

# 1. Check working directory
print(f"📁 Current directory: {os.getcwd()}")

# 2. Check .env file
env_file = Path('.env')
print(f"📄 .env file exists: {'✅' if env_file.exists() else '❌'}")

if env_file.exists():
    # 3. Load and check environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    pinecone_key = os.getenv("PINECONE_API_KEY", "").strip().strip('"\'')
    host_dense = os.getenv("HOST_DENSE", "").strip().strip('"\'')
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"\'')
    
    print(f"🔑 PINECONE_API_KEY: {'✅ Found' if pinecone_key else '❌ Missing'}")
    print(f"🌐 HOST_DENSE: {'✅ Found' if host_dense else '❌ Missing'}")
    print(f"🤖 GEMINI_API_KEY: {'✅ Found' if gemini_key else '❌ Missing'}")
    
    if all([pinecone_key, host_dense, gemini_key]):
        print("\n🎯 All environment variables found!")
        
        # 4. Test retrieval module
        try:
            sys.path.append('.')
            from retriever.parent_retrieval import initialize_connections
            initialize_connections()
            print("✅ RAG system initialization successful!")
            
        except Exception as e:
            print(f"❌ RAG initialization failed: {e}")
    else:
        print("\n❌ Missing environment variables!")
else:
    print("❌ .env file not found! Copy from .env.example")

print("\n" + "=" * 50)
print("🚀 Ready to run: streamlit run rag_app.py")
