import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk import WordNetLemmatizer

def lemm(data):
    '''función para lematizar el dataset, descompone las palabras para llegar a la raiz'''
    wordnet = WordNetLemmatizer()
    lemmanized = []
    for i in range(len(data)):
        lemmed = []
        words = word_tokenize(data['text'].iloc[i])
        for w in words:
            lemmed.append(wordnet.lemmatize(w))
        lemmanized.append(lemmed)

    data['lemmanized'] = lemmanized
    data['text'] = data['lemmanized'].apply(' '.join)
    data=data.drop("lemmanized",axis=1)
    return data