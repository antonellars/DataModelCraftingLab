import pandas as pd
import matplotlib as plt

def comparar_nubes_de_palabras(nube1, nube2, nube3):
    """
    Crea y muestra una imagen comparativa de tres nubes de palabras.
    Parameters:
    - nube1, nube2, nube3: Objetos WordCloud
    Returns:
    None
    """
    # Obtener las matrices de las nubes de palabras
    array1 = nube1.to_array()
    array2 = nube2.to_array()
    array3 = nube3.to_array()

    # Crear subplots para cada nube de palabras
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Configurar el primer subplot
    axs[0].imshow(array1, interpolation='bilinear')
    axs[0].axis('off')
    axs[0].set_title('Original')

    # Configurar el segundo subplot
    axs[1].imshow(array2, interpolation='bilinear')
    axs[1].axis('off')
    axs[1].set_title('Starting_point')

    # Configurar el tercer subplot
    axs[2].imshow(array3, interpolation='bilinear')
    axs[2].axis('off')
    axs[2].set_title('Project')

    # Ajustar el diseño
    plt.tight_layout()

    # Mostrar la imagen comparativa
    plt.show()

"""
Cómo usar la funcion:
   
comparar_nubes_de_palabras(ori, wc_point, wc_cleandf)

"""