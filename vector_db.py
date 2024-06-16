import faiss
import numpy as np

class VectorDatabase:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.paragraphs = []

    def add_embeddings(self, embeddings, paragraphs):
        self.index.add(embeddings)
        self.paragraphs.extend(paragraphs)

    def search(self, embedding, k=5):
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)
        D, I = self.index.search(embedding, k)
        relevant_texts = [self.paragraphs[i] for i in I[0]]
        return D, I, relevant_texts

    def reconstruct_n(self, indices):
        return np.array([self.index.reconstruct(int(idx)) for idx in indices])