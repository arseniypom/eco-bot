import numpy as np
from scipy.spatial.distance import cosine
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
from prepare_data import lemmatize  # Добавлен импорт функции lemmatize

model = hub.load("universal-sentence-encoder-multilingual-large_3")

def compute_embeddings(texts):
    return model(texts).numpy()

def find_matches(query, X, vectorizer, morph, lemmatized_paragraphs, paragraphs):
    lemmatized_query = lemmatize(query, morph)
    query_tfidf_vector = vectorizer.transform([lemmatized_query])

    tfidf_similarities = np.dot(X, query_tfidf_vector.T).toarray().flatten()

    paragraph_embeddings = compute_embeddings(lemmatized_paragraphs)
    query_embedding = compute_embeddings([lemmatized_query])[0]

    neural_similarities = [
        1 - cosine(query_embedding, paragraph_embedding)
        for paragraph_embedding in paragraph_embeddings
    ]

    combined_similarities = [
        0.1 * tfidf_sim + 0.9 * neural_sim
        for tfidf_sim, neural_sim in zip(tfidf_similarities, neural_similarities)
    ]

    indexed_similarities = list(enumerate(combined_similarities))
    sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)
    
    return [(paragraphs[i], sim) for i, sim in sorted_similarities[:3]]
