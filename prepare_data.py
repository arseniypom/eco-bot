import json
import nltk
import pymorphy2
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка стоп-слов
nltk.download("stopwords")

def load_paragraphs(filepath="./general.json"):
    with open(filepath, "r", encoding="utf-8") as file:
        paragraphs = json.load(file)
    return paragraphs

def lemmatize(text, morph):
    words = nltk.word_tokenize(text)
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return " ".join(lemmas)

def prepare_data(paragraphs):
    morph = pymorphy2.MorphAnalyzer()

    lemmatized_paragraphs = [lemmatize(paragraph, morph) for paragraph in paragraphs]

    stop_words = set(stopwords.words('russian'))
    lemmatized_stop_words = [morph.parse(word)[0].normal_form for word in stop_words]
    vectorizer = TfidfVectorizer(stop_words=list(lemmatized_stop_words))

    X = vectorizer.fit_transform(lemmatized_paragraphs)
    return X, vectorizer, morph, lemmatized_paragraphs
