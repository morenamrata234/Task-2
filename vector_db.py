import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)

def add_vectors_to_db(texts):
    vectors = np.array(embed_model.encode(texts))
    faiss_index.add(vectors)

def query_vector_db(query):
    query_vector = np.array(embed_model.encode([query]))
    _, indices = faiss_index.search(query_vector, k=1)
    return indices[0][0]
