"""RAG System for YouTube Videos using Hugging Face Models"""

# import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering


class YouTubeRAGSystem:
    def __init__(self):
        # Using bhot zyada Lightweight models as I am using locally
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.qa_model_name = "deepset/tinyroberta-squad2"

        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.retriever = None
        self.chain = None

    def initialize_models(self):
        """Initialize Hugging Face models"""
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

        print("Loading QA model...")
        # Initialize QA pipeline
        tokenizer = AutoTokenizer.from_pretrained(self.qa_model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(self.qa_model_name)

        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=-1  # Willl use my CPU
        )

        self.llm = HuggingFacePipeline(pipeline=qa_pipeline)
        print("Models loaded successfully")

    def get_transcript(self, video_id: str) -> str:
        """Fetch transcript from YouTube video"""
        try:
            #can use Yt-dlp(actually will use that later)
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"]) #Ok so this line has some issue (get_transcript), I've been working on this since 4 hours but could'nt solve this. I think it's the API calls of YT that's limitig me
            transcript = " ".join(chunk["text"] for chunk in transcript_list)
            print(f"Transcript fetched ({len(transcript)} characters)")
            return transcript
        except TranscriptsDisabled:
            print("No captions available for this video.")
            return ""
        except Exception as e:
            print(f"Error fetching transcript: {e}")
            return ""

    def process_transcript(self, transcript: str):
        """Process transcript and create vector store"""
        if not transcript:
            return

        # break txt into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        print(f"Split into {len(chunks)} chunks")

        # Creating vector store(will learn in depth `FYP`)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        print("Vector store created successfully!")

    def build_chain(self):
        """Build the RAG chain"""
        # Custom prompt template
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant. Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            Context: {context}
            Question: {question}

            Answer:
            """,
            input_variables=['context', 'question']
        )

        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        parallel_chain = RunnableParallel({
            'context': self.retriever | format_docs,
            'question': RunnablePassthrough()
        })

        self.chain = parallel_chain | prompt | self.llm | StrOutputParser()
        print("RAG chain built successfully!")

    def ask_question(self, question: str) -> str:
        """Ask a question about the video"""
        if not self.chain:
            return "Please initialize the system first."

        try:
            return self.chain.invoke(question)
        except Exception as e:
            return f"Error: {e}"

    def run_complete_pipeline(self, video_id: str):
        """Run the complete RAG pipeline for a video"""
        print("Initializing models...")
        self.initialize_models()

        print("Fetching transcript...")
        transcript = self.get_transcript(video_id)

        if transcript:
            print("Processing transcript...")
            self.process_transcript(transcript)
            print("Building RAG chain...")
            self.build_chain()

            print("\nSystem ready! You can now ask questions about the video.")
            return True
        return False


def main():
    """Main function to run the RAG system"""
    # Initialize the system
    rag_system = YouTubeRAGSystem()

    # YT video ki ID
    video_id = "wLGrhL8V038"


    # Running complete pipeline
    if rag_system.run_complete_pipeline(video_id):
        # Interactive question answering
        print("\n" + "=" * 50)
        print("Interactive Question Answering")
        print("Type 'quit' to exit")
        print("=" * 50)

        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break

            if question:
                answer = rag_system.ask_question(question)
                print(f"\nAnswer: {answer}")
            else:
                print("Please enter a question.")


# main function
if __name__ == "__main__":
    main()


#will make the UI using StreamLit