
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample document chunks
chunks = [
    "Generative AI creates new content such as text, images, audio and code.",
    "Large Language Models understand and generate human language using trained neural networks.",
    "Python is a popular programming language used in AI and machine learning.",
    "Machine learning allows computers to learn patterns from data."
]

# Convert chunks into embeddings
chunk_embeddings = model.encode(chunks)

# User question
question = "What are Large Language Models?"

# Convert question into embedding
question_embedding = model.encode([question])

# Calculate cosine similarity
similarities = np.dot(
    chunk_embeddings,
    question_embedding[0]
) / (
    np.linalg.norm(chunk_embeddings, axis=1)
    * np.linalg.norm(question_embedding[0])
)

# Find most relevant chunk
best_index = np.argmax(similarities)

print("SIMILARITY SEARCH RESULTS")
print("-------------------------")
print("Question:", question)
print("\nMost Relevant Information:")
print(chunks[best_index])
print("\nSimilarity Score:", similarities[best_index])
