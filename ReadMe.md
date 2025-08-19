# YouTube RAG Assistant  

A tool that extracts transcripts from YouTube videos and allows you to ask questions about the content using **RAG (Retrieval-Augmented Generation)** with LangChain + HuggingFace.  

---

## How does it works  
- Fetches transcripts from YouTube videos (if available).  
- Splits text into manageable chunks.  
- Creates vector embeddings using HuggingFace.  
- Stores and retrieves embeddings with FAISS.  
- Uses HuggingFace models for **Question Answering** over video transcripts.  

---

## Note
 Having trouble with YouTubeTranscriptApi It says
 **Error fetching transcript: type object 'YouTubeTranscriptApi' has no attribute 'get_transcript'**
 For that I came up with yt-dlp. Will learn that and use it in future.

---

## Owner
- Sheryar-bit




