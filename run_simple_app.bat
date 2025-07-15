@echo off
echo Starting RAG Political Science Chatbot...
echo.
echo Opening browser at http://localhost:8502
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0"
streamlit run simple_app.py --server.port 8502

pause
