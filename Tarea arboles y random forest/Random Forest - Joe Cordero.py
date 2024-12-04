# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE LAS AMÉRICAS
FACULTAD DE INGENIERÍA Y CIENCIAS APLICADAS
INGENIERÍA DE SOFTWARE
INTELIGENCIA ARTIFICIAL 1

RANDOM FOREST
"""
# Importación de librerías necesarias para la manipulación y visualización de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga del dataset que contiene las preferencias de las personas
# El archivo debe incluir columnas relacionadas con la edad, salario y preferencia (playa o montaña)
# Fuente del dataset: https://www.kaggle.com/datasets/jahnavipaliwal/mountains-vs-beaches-preference
dataset = pd.read_csv("C:/Users/joema/IA 1/mountains_vs_beaches_preferences.csv")

# Selección de variables independientes (X) y dependiente (y)
# Variables independientes: Edad y Salario
# Variable dependiente: Preferencia (playa o montaña)
X = dataset.iloc[:, [0, 2]].values  # Selección de columnas de edad y salario
y = dataset.iloc[:, -1].values      # Selección de la última columna como la variable dependiente

# División del conjunto de datos en entrenamiento y prueba
# Se utiliza el 25% de los datos para pruebas y el 75% para entrenamiento
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Escalado de características
# El escalado normaliza las variables para mejorar el rendimiento del modelo
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  # Ajuste y transformación del conjunto de entrenamiento
X_test = sc_X.transform(X_test)        # Transformación del conjunto de prueba usando el mismo ajuste

# Creación y entrenamiento del modelo Random Forest
# El modelo se configura con 10 árboles y utiliza la entropía como criterio de decisión
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
rf.fit(X_train, y_train)  # Entrenamiento del modelo con los datos de entrenamiento

# Predicción de resultados en el conjunto de prueba
y_pred = rf.predict(X_test)

# Evaluación del modelo mediante una matriz de confusión
from sklearn.metrics import confusion_matrix
conmat = confusion_matrix(y_test, y_pred)

# Visualización de los resultados en el conjunto de entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

plt.contourf(X1, X2, rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Bosques Aleatorios (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Salario')
plt.legend()
plt.show()

# Visualización de los resultados en el conjunto de prueba
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

plt.contourf(X1, X2, rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Bosques Aleatorios (Conjunto de Prueba)')
plt.xlabel('Edad')
plt.ylabel('Salario')
plt.legend()
plt.show()