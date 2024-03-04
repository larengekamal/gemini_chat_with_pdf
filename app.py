from dotenv import load_dotenv
import streamlit as st
import os 
import io
import google.generativeai as genai 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv() ## loading all the environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(uploaded_files):
    texts = ""
    for file in uploaded_files:
        reader = PdfReader(file)
        text = ""
        for page_number in range(len(reader.pages)):
            text += reader.pages[page_number].extract_text()
        texts+=text
    return texts

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("4. Saved STORE "+str(len(text_chunks)))
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    
def get_conversational_chain():
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in the
    provided context just say, "answer is not available in the context" don't provide the wrong answer.
    Context:\n{context}?\n
    Question:\n{question}\n
    
    Answer:
    """    
    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index",embeddings=embeddings)
    docs=new_db.similarity_search(user_question)
    
    chain=get_conversational_chain()
    response=chain({"input_documents":docs, "question":user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply:", response["output_text"])
    
    
def main():
    st.set_page_config(page_title="Chat with PDFs")
    st.header('Chat with PDFs | Gemini LLM')
    input=st.text_input('Ask a Question from PDF Files: ', key='input')
    if(input):
        user_input(input)
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your PDF Files and Click on the Button Submit and Process", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit and Process"):
            with st.spinner("Processing.."):
                raw_text=get_pdf_text(pdf_docs)
                print("1. GOT RAW TEXT "+raw_text)
                text_chunks=get_text_chunks(raw_text)
                print("2. GOT TEXT CHUNKS")
                get_vector_store(text_chunks)
                st.success("Done")
            
            
if __name__ == "__main__":
    main()