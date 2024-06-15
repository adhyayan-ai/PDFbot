from embedder import model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def verify_response(response, db):
    response_embedding = model.encode([response])
    response_embedding = np.array(response_embedding).reshape(1, -1)
    print("Response Embedding:", response_embedding)

    distances, indices, relevant_texts = db.search(response_embedding)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Relevant Texts:", relevant_texts)

    reconstructed_embeddings = db.reconstruct_n(indices[0])
    print("Reconstructed Embeddings:", reconstructed_embeddings)
    similarities = cosine_similarity(response_embedding, reconstructed_embeddings)
    print("Cosine Similarities:", similarities)

    mean_similarity = np.mean(similarities[0])
    std_similarity = np.std(similarities[0])
    dynamic_threshold = mean_similarity + 0.5 * std_similarity
    print("Mean Similarity:", mean_similarity)
    print("Standard Deviation of Similarity:", std_similarity)
    print("Dynamic Threshold:", dynamic_threshold)

    verified = any(sim >= dynamic_threshold for sim in similarities[0])
    print("Verified:", verified)
    return verified