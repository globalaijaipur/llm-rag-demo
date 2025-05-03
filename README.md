# LLM & RAG Demo with LangChain and LLAMA2

This repository contains a quick demonstration of Large Language Models (LLMs) and Retrieval Augmented Generation (RAG) using Python and LangChain. The project includes two different implementations showcasing basic LLM integration and RAG capabilities.

## Project Structure

```
├── 1.py         # Simple LLM chat implementation with Streamlit UI
├── 2.py         # RAG implementation with PDF document processing
└── speech.pdf   # Sample PDF document for RAG demo
```

## Implementation Details

### 1. Simple LLM Chat (`1.py`)

This implementation demonstrates a basic chat interface using:
- Streamlit for the web interface
- LangChain for LLM orchestration
- Ollama for local LLM inference using LLAMA2
- Environment variable configuration for LangChain tracing

Key features:
- Simple chat prompt template
- Web-based user interface
- Integration with local LLAMA2 model via Ollama
- Structured output parsing

### 2. RAG Implementation (`2.py`)

This file showcases a complete RAG pipeline with the following components:

- PDF document loading and processing
- Text chunking with recursive character splitting
- Vector embeddings using HuggingFace's sentence transformers
- Vector storage using FAISS
- Question-answering chain implementation
- Interactive command-line interface

Technical components:
- Document Loading: `PyPDFLoader`
- Text Splitting: `RecursiveCharacterTextSplitter`
- Embeddings: `HuggingFaceEmbeddings` (using "sentence-transformers/all-MiniLM-L6-v2")
- Vector Store: `FAISS`
- LLM: Ollama (LLAMA2)
- Chain Type: RetrievalQA with "stuff" strategy

## Prerequisites

- Python 3.x
- Ollama installed and running locally
- Required Python packages:
  - langchain
  - langchain-ollama
  - streamlit
  - python-dotenv
  - sentence-transformers
  - faiss-cpu
  - PyPDF2

## Usage

1. For the chat interface:
```bash
streamlit run 1.py
```

2. For the RAG demo:
```bash
python 2.py
```
Make sure you have a PDF file named "speech.pdf" in the same directory when running the RAG demo.

## Technical Details

### Vector Store Configuration
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Top-k retrieval: 4 chunks per query

### Model Configuration
- LLM: LLAMA2 (via Ollama)
- Embedding Model: all-MiniLM-L6-v2
- Default Ollama endpoint: http://localhost:11434

## Educational Value

This project serves as an excellent introduction to:
1. Basic LLM integration
2. RAG architecture and implementation
3. Vector databases and embeddings
4. Document processing and chunking
5. Question-answering systems
6. LangChain's capabilities and patterns

Perfect for developers looking to understand the fundamentals of modern LLM applications and RAG systems.
