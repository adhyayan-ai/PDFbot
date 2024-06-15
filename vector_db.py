import faiss
import numpy as np

class VectorDatabase:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.sentences = []

    def add_embeddings(self, embeddings, sentences):
        if len(embeddings) == 0:
            print("No embeddings to add.")
            return
        self.index.add(embeddings)
        self.sentences.extend(sentences)

    def search(self, embedding, k=5):
        if embedding.ndim == 1:
            embedding = np.expand_dims(embedding, axis=0)
        D, I = self.index.search(embedding, k)
        print("Search Distances:", D)  # Debugging statement
        print("Search Indices:", I)  # Debugging statement
        relevant_texts = [self.sentences[i] for i in I[0]]
        return D, I, relevant_texts

    def reconstruct_n(self, indices):
        return np.array([self.index.reconstruct(int(idx)) for idx in indices])