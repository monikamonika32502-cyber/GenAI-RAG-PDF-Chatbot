
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample text
sentences = [
    "Generative AI creates new content.",
    "Large Language Models understand human language.",
    "Python is a programming language."
]

# Generate embeddings
embeddings = model.encode(sentences)

print("EMBEDDING RESULTS")
print("-----------------")

for i, embedding in enumerate(embeddings):
    print(f"\nSentence {i + 1}: {sentences[i]}")
    print("Vector dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])
