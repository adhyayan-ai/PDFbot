from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

if api_key is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=api_key)

def select_relevant_context(query_embedding, paragraphs, paragraph_embeddings, max_tokens=2000):
    # Calculate cosine similarity between the query and each paragraph
    similarities = cosine_similarity(query_embedding, paragraph_embeddings)
    
    # Sort paragraphs by similarity score in descending order
    sorted_indices = np.argsort(similarities[0])[::-1]
    
    selected_paragraphs = []
    total_tokens = 0
    for idx in sorted_indices:
        paragraph = paragraphs[idx]
        paragraph_tokens = len(paragraph.split())
        if total_tokens + paragraph_tokens > max_tokens:
            break
        selected_paragraphs.append(paragraph)
        total_tokens += paragraph_tokens
    
    return selected_paragraphs

def generate_response(query, query_embedding, paragraphs, paragraph_embeddings):
    relevant_context = select_relevant_context(query_embedding, paragraphs, paragraph_embeddings)
    context_str = " ".join(relevant_context)
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{query}\n\nContext: {context_str}"}
        ]
    )
    return response.choices[0].message.content