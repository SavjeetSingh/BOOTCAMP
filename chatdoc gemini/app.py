import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from io import BytesIO
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_stream = BytesIO(pdf.read())
        pdf_reader = PdfReader(pdf_stream)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in the provided context, just say, "The answer is not available in the context". Do not provide incorrect information.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    #st.write("Retrieved Documents:", docs)
    
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    st.write("Chain Response:", response)
    st.write("Reply:", response["output_text"])

def main():
    st.set_page_config(page_title="Chat Gemini")
    st.header("Chat Gemini")
    user_question = st.text_input("Ask a question from the PDF files")
    
    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Proceed"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    #st.write("Raw Text:", raw_text[:500])  # Show first 500 characters for brevity
                    text_chunks = get_text_chunks(raw_text)
                    #st.write("Text Chunks:", text_chunks[:5])  # Show first 5 chunks for brevity
                    get_vectorstore(text_chunks)
                    st.success("Done")

if __name__ == "__main__":
    main()
