
# 🤖 RAG-Based PDF Chatbot Using Generative AI

## Project Overview
This project is a Retrieval-Augmented Generation (RAG) based PDF chatbot. It allows users to upload PDF documents and ask questions about their contents.

The system retrieves relevant information from the uploaded PDF using text embeddings and similarity search. Google Gemini generates answers based on the retrieved information.

## Technologies Used
- Python
- Streamlit
- Google Gemini API
- Sentence Transformers
- NumPy
- PyPDF

## Features
1. PDF document upload
2. Automatic text extraction
3. Text chunking
4. Text embedding generation
5. Semantic similarity search
6. Relevant information retrieval
7. AI-powered answer generation
8. Interactive Streamlit interface

## Project Workflow
PDF Upload → Text Extraction → Chunking → Embeddings → Similarity Search → Information Retrieval → Gemini Response

## Project Structure
```text
GenAI_Capstone_Project/
│
├── app.py
├── rag.py
├── embeddings.py
├── gemini_test.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
└── sample.pdf
```

## Installation

1. Clone the repository:
   git clone YOUR_GITHUB_REPOSITORY_URL

2. Install dependencies:
   pip install -r requirements.txt

3. Create a `.env` file and add your Gemini API key:
   GEMINI_API_KEY=your_api_key_here

4. Run the application:
   streamlit run app.py

## Expected Output
Users can upload a PDF, ask questions, and receive AI-generated answers based on relevant document content.

## Future Enhancements
- Support multiple PDF documents
- Add chat history
- Improve document retrieval
- Support additional document formats

## Author
GenAI Capstone Project
