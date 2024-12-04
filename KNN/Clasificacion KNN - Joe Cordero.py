# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE LAS AMÉRICAS
FACULTAD DE INGENIERÍA Y CIENCIAS APLICADAS
INGENIERÍA DE SOFTWARE
INTELIGENCIA ARTIFICIAL 1

ANÁLISIS Y CLASIFICACIÓN DE DATOS CON K-NEAREST NEIGHBORS (K-NN)
POR: JOE CORDERO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga del dataset
# El dataset consiste en varias variables médicas predictoras (independientes) y una variable objetivo (dependiente), `Outcome`. 
# Las variables independientes incluyen el número de embarazos, el índice de masa corporal (BMI), nivel de insulina, edad, entre otros. 
# La variable `Outcome` indica si el paciente presenta diabetes (1) o no (0).
# Fuente: https://www.kaggle.com/code/mbalvi75/08-knn-diabetes-dataset/input
dataset = pd.read_csv("./IA 1/diabetes.csv")

# Selección de variables independientes (columnas 1 y 5) y dependiente (última columna)
X = dataset.iloc[:, [1, 5]]  # Variables independientes seleccionadas: Glucose y BMI
y = dataset.iloc[:, -1]      # Variable dependiente: Outcome (diagnóstico de diabetes)

# Imputación de valores faltantes utilizando la mediana como estrategia
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy="median")
imputer = imputer.fit(X)
X = imputer.transform(X)

# División del dataset en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Escalado de las variables para normalizar las características
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Configuración del clasificador K-Nearest Neighbors (K-NN)
from sklearn.neighbors import KNeighborsClassifier
##p=1 minkowsi
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)  #p=1 minkowsi p=2 Distancia Euclidiana
knn.fit(X_train, y_train)

# Predicción de resultados utilizando el conjunto de prueba
y_pred = knn.predict(X_test)

# Construcción de la matriz de confusión para evaluar el rendimiento del clasificador
from sklearn.metrics import confusion_matrix
confusion_matrix_result = confusion_matrix(y_test, y_pred)

# Visualización gráfica del modelo aplicado al conjunto de entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title("K-NN (Conjunto de entrenamiento)")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.legend()
plt.show()

# Visualización gráfica del modelo aplicado al conjunto de prueba
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title("K-NN (Conjunto de prueba)")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.legend()
plt.show()

Objetivo: Clasificar observaciones basado en la clase de sus vecinos más cercanos.
Ventajas: Sencillo, no requiere suposiciones sobre la distribución de los datos.
Concepto clave: Mide distancias (Euclidiana, Manhattan) para determinar las clases.
Escalado: Necesario.
Ideal cuando: Hay muchas observaciones y relaciones complejas, pero es sensible al ruido en los datos.
KNN se usó porque el análisis requería un enfoque no paramétrico y basado en distancias para clasificar instancias según sus vecinos más cercanos. Esto es útil cuando la relación entre las variables no es lineal y se quiere aprovechar la simplicidad de KNN.
Dataset: Diagnóstico de diabetes basado en variables como Glucosa y BMI.
Ventajas específicas:
No hace suposiciones sobre la distribución de los datos.
Útil para datos con relaciones no lineales entre variables.
