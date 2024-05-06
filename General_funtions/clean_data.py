from unidecode import unidecode
import nltk
from nltk import WordNetLemmatizer
from wordcloud import WordCloud
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')
import re

def cleaning_starting_point(text):


    """Función encontrada en Kaggel, se utiliza para comparar performances"""
    text = text.lower()
    text =  re.sub(r'@\S+', '',text)  # remove twitter handles
    text =  re.sub(r'http\S+', '',text) # remove urls
    text =  re.sub(r'pic.\S+', '',text)
    text =  re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚ']", ' ',text) # only keeps characters
    text =  re.sub(r'\s+[a-zA-ZáéíóúÁÉÍÓÚ]\s+', ' ', text+' ')  # keep words with length>1 only
    text = "".join([i for i in text if i not in string.punctuation])
    words = word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')   # remove stopwords
    text = " ".join([i for i in words if i not in stopwords])
    text= re.sub("\s[\s]+", " ",text).strip()
    text= re.sub("\s[\s]+", " ",text).strip() # remove repeated/leading/trailing spaces
    return text

def clean_text_DF(df):
    '''Limpia el texto en la columna "text" del DataFrame df, 
    convirte texto a minúsculas, se eliminan simbolos, letras sueltas, stopwords" en inglés,
    números y URLs, tokenizando las palabras,  
    Retorna un nuevo DataFrame con la columna "text" limpia.'''
    
    clean_df = df.copy(deep=True)
    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    def post_processs_words(text: str) -> str:
        words = word_tokenize(text)
        # Eliminar letras suelta y palabras compuestas por 2 letras
        words = [word for word in words if len(word) > 1 and len(word) > 2]
        text = " ".join([i for i in words if i not in stop_words])
        return text
    
    clean_df["text"] = clean_df["text"].apply(lambda x: x.lower())             \
        .apply(lambda x: re.sub(r"[,.\"!@#$%^&*(){}?/;¡¿´`~:'<>+=-]", "", x))  \
        .apply(lambda x: re.sub(r'\d+', '', x))                                \
        .apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))              \
        .apply(lambda x: re.sub("\s+"," ",x))                                  \
        .apply(lambda x: unidecode(x))                                         \
        .apply(post_processs_words)          
    #print(clean_df.head())
    
    return clean_df