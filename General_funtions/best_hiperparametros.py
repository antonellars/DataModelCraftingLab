from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def best_hiperparametros(x_train, y_train):
    # Definir el diccionario de hiperparámetros a explorar
    param_grid = {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # Inicializar el objeto GridSearchCV
    grid_search = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

    # Ajustar el modelo a los datos de entrenamiento
    grid_search.fit(x_train, y_train)

    # Imprimir los mejores hiperparámetros encontrados
    print("Mejores Hyperparámetros:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Devolver los mejores hiperparámetros encontrados
    return grid_search.best_params_

'''Utilizar la función con los datos de entrenamiento x_train y las etiquetas y_train

Cómo llamar a la función

mejores_hiperparametros = best_hiperparametros(x_train, y_train)

'''