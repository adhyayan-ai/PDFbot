from prompt_toolkit import PromptSession
from pdf_parser import parse_pdf
from embedder import get_embeddings
from vector_db import VectorDatabase
from chatbot import generate_response
from verifier import verify_response
import os

def main():
    file_path = 'document.pdf'
    
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return
    
    pdf_text = parse_pdf(file_path)
    sentences, embeddings = get_embeddings(pdf_text)
    db = VectorDatabase()
    db.add_embeddings(embeddings, sentences)
    
    session = PromptSession()
    while True:
        query = session.prompt('You: ')
        if query.lower() in ['exit', 'quit']:
            break
        response = generate_response(query)
        if verify_response(response, db):
            print("Chatbot: ", response)
        else:
            print("Chatbot: Sorry, I couldn't verify the information. Please try again.")

if __name__ == '__main__':
    main()