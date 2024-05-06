from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_pipelines_mx(X, y, X_test, y_test, text_lr, text_lr2):
    ''' La función devuelve el DataFrame de resultados, así como las matrices de confusión
    asociadas a cada modelo, permitiendo una evaluación completa del rendimiento de los modelos
    con los diferentes pipelines '''
    
    # Ajustar los modelos
    text_lr.fit(X, y)
    text_lr2.fit(X, y)
    
    # Hacer predicciones en el conjunto de prueba
    y_pred = text_lr.predict(X_test)
    y_pred2 = text_lr2.predict(X_test)
    
    #Calcular matrix de confusion
    conf_matrix_lr = confusion_matrix(y_test, y_pred)
    conf_matrix_lr2 = confusion_matrix(y_test, y_pred2)
    
    # Calcular métricas para el primer modelo
    precision1 = precision_score(y_test, y_pred)
    recall1 = recall_score(y_test, y_pred)
    f11 = f1_score(y_test, y_pred)
    accuracy1 = accuracy_score(y_test, y_pred)
    mse1 = mean_squared_error(y_test, y_pred)

    # Calcular métricas para el segundo modelo
    precision2 = precision_score(y_test, y_pred2)
    recall2 = recall_score(y_test, y_pred2)
    f12 = f1_score(y_test, y_pred2)
    accuracy2 = accuracy_score(y_test, y_pred2)
    mse2 = mean_squared_error(y_test, y_pred2)

    # Crear un DataFrame con los resultados
    results = pd.DataFrame({
        'Modelo': ['Pipeline Simple', 'Pipeline con TruncateSVD'],
        'Precision': [precision1, precision2],
        'Recall': [recall1, recall2],
        'F1': [f11, f12],
        'Accuracy': [accuracy1, accuracy2],
        'MSE': [mse1, mse2]
    })

    # Imprimir el cuadro comparativo
    #print("Cuadro Comparativo: Pipeline Simple vs Pipeline con TruncateSVD")
    return results, conf_matrix_lr, conf_matrix_lr2