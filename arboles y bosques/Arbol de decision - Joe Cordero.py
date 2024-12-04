# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE LAS AMÉRICAS
FACULTAD DE INGENIERÍA Y CIENCIAS APLICADAS
INGENIERÍA DE SOFTWARE
INTELIGENCIA ARTIFICIAL 1

ARBÓL DE DECISIÓN
"""
# Importación de librerías necesarias para la manipulación y visualización de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga del dataset que contiene las preferencias de las personas
# El archivo incluye columnas relacionadas con la edad, salario y preferencia (playa o montaña)
# Fuente: https://www.kaggle.com/datasets/jahnavipaliwal/mountains-vs-beaches-preference
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

# Escalado variables
##Esto permite que las variables sean comparables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  # Ajuste y transformación del conjunto de entrenamiento
X_test = sc_X.transform(X_test)        # Transformación del conjunto de prueba utilizando el mismo ajuste

# Creación y entrenamiento del Árbol de Decisión
# Se utiliza la entropía como criterio de división en lugar del índice de Gini (que es el predeterminado)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy", random_state=0)
dt.fit(X_train, y_train)  # Entrenamiento del modelo con los datos de entrenamiento

# Predicción de resultados en el conjunto de prueba
y_pred = dt.predict(X_test)

# Evaluación del rendimiento del modelo utilizando una matriz de confusión
# Esto ayuda a medir la precisión de las predicciones realizadas por el modelo
from sklearn.metrics import confusion_matrix
conmat = confusion_matrix(y_test, y_pred)

# Visualización del modelo en el conjunto de entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

plt.contourf(X1, X2, dt.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Árbol de decisión (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Salario')
plt.legend()
plt.show()

# Visualización del modelo en el conjunto de prueba
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

plt.contourf(X1, X2, dt.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Árbol de decisión (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Salario')
plt.legend()
plt.show()


Objetivo: Crear un modelo jerárquico para clasificar datos basado en preguntas binarias.
Ventajas: Fácil de interpretar y no requiere escalado.
Concepto clave: Utiliza medidas como Gini o Entropía para dividir datos.
Escalado: No necesario.
Ideal cuando: Se buscan reglas claras para explicar las decisiones.

Justificación:
Se utilizó este método porque el análisis requería un modelo explicativo, donde las reglas de clasificación (por ejemplo, preferencia por playa o montaña basada en Edad y Salario) fueran claras y fácilmente interpretables.
Dataset: Clasificación basada en preferencias personales.
Ventajas específicas:
No requiere escalado de datos.
Genera reglas comprensibles y visuales para tomar decisiones.
