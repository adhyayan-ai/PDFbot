from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_embeddings(text):
    sentences = text.split('\n')
    embeddings = model.encode(sentences)
    return sentences, embeddings