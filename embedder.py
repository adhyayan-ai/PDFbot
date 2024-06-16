from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_embeddings(text):
    if isinstance(text, list):
        embeddings = model.encode(text)
        return text, embeddings
    else:
        sentences = text.split('\n')
        embeddings = model.encode(sentences)
        return sentences, embeddings