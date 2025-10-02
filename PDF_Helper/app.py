# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
#
#
# load_dotenv()
#
# import os
# os.environ["GEMINI_API_KEY"] = "AIzaSyDug62DN520ZYE75ArQgG1Zvin1__HZt_A"
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#
# #This function is responsible for PDF reading
# def get_pdf_txt(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text
#
#
#
# def get_txt_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks=text_splitter.split_text(text)
#     return chunks
#
# def get_vector_store(text_chunks):
#     embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
#     vector_store.save_local("faiss_index")
#
# def get_conversational_chain():
#     prompt_template= """
#     Answer the question as detailed as possible from the provided context, make sure
#     to provide all the details, if the answer is not in the provided context just say,
#     "Answer is not available in the context", don't provide the wrong answer.
#     Context:\n {context}?\n
#     Question:\n {question}?\n
#
#     Answer:
#     """
#
#     model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt= PromptTemplate(template=prompt_template, input_variables=["context","question"])
#     chain= load_qa_chain(model,chain_type="stuff",prompt=prompt)
#     return chain
#
#
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#
#     new_db = FAISS.load_local("faiss_index", embeddings)
#     docs = new_db.similarity_search(user_question)
#
#     chain = get_conversational_chain()
#
#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True
#     )
#
#     print(response)
#     st.write("Reply:", response["output_text"])
#
# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with multiple PDFs")
#
#     user_question = st.text_input("Ask a Question from the PDF Files")
#
#     if user_question:
#         user_input(user_question)
#
#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
#                                     accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_txt(pdf_docs)
#                 text_chunks = get_txt_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")
#
#
# if __name__ == "__main__":
#     main()


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings  # Alternative: sentence-transformers
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq  # Using Groq for LLM
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

os.getenv("GROQ_API_KEY")
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )


# This function is responsible for PDF reading
def get_pdf_txt(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_txt_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say: 
    "Answer is not available in the context". Don't make up answers.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    try:
        embeddings = get_embeddings()


        if not os.path.exists("faiss_index"):
            st.error("Please process PDF files first before asking questions.")
            return

        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question, k=3)  # Gets top 3 relevant chunks

        chain = get_conversational_chain()

        with st.spinner("Thinking..."):
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

        st.write("**Reply:**", response["output_text"])


        with st.expander("View relevant sources"):
            for i, doc in enumerate(docs):
                st.write(f"**Source {i + 1}:**")
                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")


def main():
    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon="📄",
        layout="wide"
    )

    st.title("Chat with Multiple PDFs (Developed By Sheryar)")
    st.markdown("Upload your PDF files and ask questions about their content")


    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None


    col1, col2 = st.columns([2, 1])

    with col1:
        user_question = st.text_input(
            "Ask a question about your PDFs:",
            placeholder="What is the main topic of the document?"
        )

        if user_question:
            user_input(user_question)

    with col2:
        st.info("""
        **How to use:**
        1. Upload PDF files in the sidebar
        2. Click 'Process PDFs'
        3. Ask questions about your documents
        """)

    # Sidebar
    with st.sidebar:
        st.header("Document Processing")

        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True,
            type="pdf",
            help="Select one or more PDF files to process"
        )

        if st.button("Process PDFs", type="primary", use_container_width=True):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return

            with st.spinner("Processing PDFs... This may take a while depending on file sizes."):
                try:
                    # Extract text
                    raw_text = get_pdf_txt(pdf_docs)

                    if not raw_text.strip():
                        st.error("""
                        No text could be extracted from the PDF files. 
                        This might happen with:
                        - Scanned PDFs (images)
                        - Protected PDFs
                        - Empty files
                        """)
                        return

                    st.info(f"Extracted {len(raw_text)} characters from {len(pdf_docs)} PDF(s)")

                    # Split into chunks
                    text_chunks = get_txt_chunks(raw_text)
                    st.info(f"Created {len(text_chunks)} text chunks")

                    # Create vector store
                    get_vector_store(text_chunks)
                    st.session_state.processed = True

                    st.success("✅ PDFs processed successfully!")
                    st.balloons()

                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")


        if st.button("🗑️ Clear Cache", use_container_width=True):
            if os.path.exists("faiss_index"):
                import shutil
                shutil.rmtree("faiss_index")
            st.session_state.processed = False
            st.success("Cache cleared!")




if __name__ == "__main__":
    main()