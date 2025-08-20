# RAG Practice Projects

A collection of Retrieval-Augmented Generation (RAG) projects using LangChain, HuggingFace, and various data sources.

## Projects Included

### 1. YouTube RAG Assistant
A tool that extracts content from YouTube videos and allows you to ask questions about the content using RAG.

**Features:**
- Extracts transcripts from YouTube videos using yt-dlp
- Splits content into manageable chunks
- Creates vector embeddings using HuggingFace
- Stores and retrieves embeddings with FAISS/ChromaDB
- Uses HuggingFace models for question answering

**Status:** Active but need some debugging YT API calls are being blocked (Update with yt-dlp integration)

### 2. PDF Document RAG System
A robust system for querying information from PDF documents using RAG methodology.

**Features:**
- PDF text extraction and processing
- Chunking with configurable overlap
- Vector storage with ChromaDB
- Multiple LLM backend support (Ollama, HuggingFace, OpenAI)
- Source citation and context retrieval

**Status:** Active (Jupyter Notebook based)

## 🛠️ Tech Stack

- **Framework:** LangChain, LangChain Community
- **LLMs:** HuggingFace, Ollama, OpenAI
- **Vector Stores:** ChromaDB, FAISS
- **Embeddings:** FastEmbed, HuggingFace Embeddings
- **Document Loaders:** PyPDF, YouTube (yt-dlp)
- **Text Processing:** Recursive Text Splitting


### For PDF Document RAG:
```python
from document_rag import PDFRAGSystem

rag = PDFRAGSystem("your_document.pdf")
rag.ask("What is the main topic of this document?")
```

### For YouTube RAG:
```python
from youtube_rag import YouTubeRAGAssistant

rag = YouTubeRAGAssistant("https://youtube.com/watch?v=example")
rag.ask("What are the key points discussed in this video?")
```

**Maintained by:** Sheryar-bit  
**Owner:** [Sheryar-bit](https://github.com/Sheryar-bit)