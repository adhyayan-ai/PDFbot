from prompt_toolkit import PromptSession
from pdf_parser import parse_pdf
from embedder import get_embeddings
from vector_db import VectorDatabase
from chatbot import generate_response
from verifier import verify_response
import os
import numpy as np

def main():
    file_path = 'document.pdf'
    
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return
    
    pdf_text = parse_pdf(file_path)
    paragraphs, paragraph_embeddings = get_embeddings(pdf_text)
    db = VectorDatabase()
    db.add_embeddings(paragraph_embeddings, paragraphs)
    
    session = PromptSession()
    while True:
        query = session.prompt('You: ')
        if query.lower() in ['exit', 'quit']:
            break
        query_embedding = np.array(get_embeddings(query)[1]).reshape(1, -1)
        response = generate_response(query, query_embedding, paragraphs, paragraph_embeddings)
        if verify_response(response, db):
            print("Chatbot: ", response)
        else:
            print("Chatbot: Sorry, I couldn't verify the information. Please try again.")

if __name__ == '__main__':
    main()