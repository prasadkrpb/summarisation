import os
import streamlit as st
import langchain
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pypdf import PdfReader
from rouge import Rouge

from dotenv import load_dotenv
load_dotenv()

# Set up LLM with Groq API
api_key = os.getenv("GROQ_API_KEY")
#llama_model = ChatGroq(groq_api_key=api_key,model="llama3-8b-8192")
#llama_model = ChatGroq(groq_api_key=api_key,model="llama3-70b-8192")
llama_model = ChatGroq(groq_api_key=api_key,model="gemma2-9b-it")



# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to summarize text
def summarize_text(document_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # ✅ Convert text into Document objects
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(document_text)]
    
    chain = load_summarize_chain(
        llm=llama_model,
        chain_type="map_reduce"
    )
    
    return chain.run(docs)  # ✅ Pass list of Documents instead of a string

# Function to validate summary
def evaluate_summary(predicted, reference):
    rouge = Rouge()
    scores = rouge.get_scores(predicted, reference)
    return scores

if __name__ == "__main__":
    pdf_path = "sample.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            document_text = extract_text_from_pdf(pdf_file)
            summary = summarize_text(document_text)
            print("Summary:\n", summary)
