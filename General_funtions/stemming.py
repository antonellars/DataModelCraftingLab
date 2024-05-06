import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def stemming_dataset(dataset):
    
    """
    Aplica el proceso de "stemming" a un conjunto de datos textual.

    Parámetros:
    - dataset: DataFrame de pandas. Debe contener una columna llamada 'text' con los textos a stemmizar.

    Retorna:
    - dataset_stemmizado: DataFrame de pandas. El conjunto de datos con los textos stemmizados.
    """
    
    nltk.download('punkt')  # Descargar el tokenizador de palabras
    stemmer = PorterStemmer()
    dataset['text'] = dataset['text'].apply(lambda text: ' '.join([stemmer.stem(word) for word in word_tokenize(text)]))
    return dataset