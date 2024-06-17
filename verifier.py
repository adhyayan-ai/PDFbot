from embedder import model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def verify_response(response, db):
    response_embedding = model.encode([response])
    response_embedding = np.array(response_embedding).reshape(1, -1)
    distances, indices, relevant_texts = db.search(response_embedding)

    reconstructed_embeddings = db.reconstruct_n(indices[0])
    similarities = cosine_similarity(response_embedding, reconstructed_embeddings)
    mean_similarity = np.mean(similarities[0])
    std_similarity = np.std(similarities[0])
    
    dynamic_threshold = mean_similarity + 0.5 * std_similarity

    verified = any(sim >= dynamic_threshold for sim in similarities[0])
    return verified