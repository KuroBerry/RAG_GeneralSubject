@echo off
echo Testing RAG App Environment...
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version
echo.

echo Checking environment variables...
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('PINECONE_API_KEY:', 'Found' if os.getenv('PINECONE_API_KEY') else 'Not found'); print('HOST_DENSE:', 'Found' if os.getenv('HOST_DENSE') else 'Not found'); print('GEMINI_API_KEY:', 'Found' if os.getenv('GEMINI_API_KEY') else 'Not found')"
echo.

echo Testing retrieval module...
python -c "from retriever.parent_retrieval import initialize_connections; initialize_connections(); print('✅ Initialization successful!')"
echo.

echo Starting RAG App...
streamlit run rag_app.py --server.port 8503

pause
