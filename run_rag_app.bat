@echo off
echo 🎓 Starting RAG Political Science Chatbot...
echo.
echo 🔧 Attempting to initialize Parent Document Retrieval...
echo 💻 Opening at http://localhost:8503
echo.
echo ⏹️  Press Ctrl+C to stop
echo.

cd /d "%~dp0"
streamlit run rag_app.py --server.port 8503

pause
