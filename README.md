# PDF AI Assistant

An AI-powered assistant that lets you upload PDFs, ask questions about them, and chat with a general LLM.  
Built with FastAPI, Groq Llama3-70B, and ChromaDB for efficient document retrieval.  

---

# Features

~ Load environment variables from `.env` (no hardcoding API keys).  
~ Automatically creates `pdf/` and `db/` folders.  
~ Upload PDFs → parse text → split into chunks (~1024 chars with 80 overlap).  
~ Store embeddings in ChromaDB (persistent vector database).  
~ Query your PDFs with semantic search (retriever with top-k=20, score threshold=0.1).  
~ Ask general LLM questions (outside of PDFs).  
~ REST API endpoints for easy integration.  

---


