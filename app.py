import streamlit as st
import langchain
from main import extract_text_from_pdf, summarize_text

st.title("📄 Research Paper Summarizer")

# File uploader
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.write("Processing file...")

    # Extract text
    document_text = extract_text_from_pdf(uploaded_file)
    
    # Summarize
    summary = summarize_text(document_text)
    
    # Display summary
    st.subheader("Summary:")
    st.write(summary)
    
    # Option to download
    st.download_button("Download Summary", summary, file_name="summary.txt")
