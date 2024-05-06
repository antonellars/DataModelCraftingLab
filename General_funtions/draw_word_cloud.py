# TO INSTALL THE LIBRARY CLOUD USE -> !pip install wordcloud
import pandas as np
from wordcloud import WordCloud
from PIL import Image

def Word_Cloud(df, mask_path = None):
    """
    Genera y muestra una nube de palabras a partir de un DataFrame de pandas.
    Parameters:
    - df: DataFrame de pandas que debe contener una columna llamada 'text' con el texto.
    - mask_path: Ruta de la máscara de la nube de palabras (formato de imagen).
    Returns: wordcloud object
    """
    texto_completo = ' '.join(df['text'])

    if mask_path:
        # Cargar la máscara
        mask = np.array(Image.open(mask_path))
        
        # Crear la nube de palabras con la máscara
        wordcloud = WordCloud(width=300, height=300, random_state=21, max_font_size=110, mask=mask, background_color= 'black', contour_color= 'white', contour_width = 0.001).generate(texto_completo)
    else:
        # Crear la nube de palabras sin máscara
        wordcloud = WordCloud(width=300, height=300, random_state=21, max_font_size=110).generate(texto_completo)

    # Visualizar la nube de palabras
    '''
    plt.figure(figsize=(6, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Movie Word Cloud')
    plt.show()
    '''
    return wordcloud