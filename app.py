
import streamlit as st
import os
import numpy as np

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG-Based PDF Chatbot")
st.write("Upload a PDF and ask questions based on its content.")

if not api_key:
    st.error("Gemini API key not found. Check your .env file.")
    st.stop()

# Initialize Gemini and embedding model
client = genai.Client(api_key=api_key)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# Split text into chunks
def split_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []

    for start in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks

# Find relevant chunks
def retrieve_relevant_chunks(question, chunks, embeddings, top_k=3):
    question_embedding = model.encode([question])[0]

    similarities = np.dot(
        embeddings,
        question_embedding
    ) / (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(question_embedding)
        + 1e-10
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        (chunks[i], float(similarities[i]))
        for i in top_indices
    ]

# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

if uploaded_file:
    reader = PdfReader(uploaded_file)

    document_text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    if not document_text.strip():
        st.error("No readable text found in this PDF.")
        st.stop()

    chunks = split_text(document_text)

    st.success("PDF uploaded and processed successfully!")
    st.write("Total pages:", len(reader.pages))
    st.write("Total chunks:", len(chunks))

    # Generate chunk embeddings
    with st.spinner("Creating document embeddings..."):
        embeddings = model.encode(chunks)

    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if st.button("Get Answer"):
        if question.strip():

            with st.spinner("Searching document and generating answer..."):

                relevant_chunks = retrieve_relevant_chunks(
                    question,
                    chunks,
                    embeddings
                )

                context = "\n\n".join(
                    chunk for chunk, score in relevant_chunks
                )

                prompt = f"""
Answer the question using only the provided document context.

If the answer is not available in the context,
clearly say that the document does not contain the information.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}
"""

                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

            st.subheader("AI Answer")
            st.write(response.text)

            with st.expander("View retrieved document chunks"):
                for i, (chunk, score) in enumerate(relevant_chunks):
                    st.write(f"Chunk {i + 1}")
                    st.write("Similarity:", round(score, 4))
                    st.write(chunk)
        else:
            st.warning("Please enter a question.")
