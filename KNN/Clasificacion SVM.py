# -*- coding: utf-8 -*-
"""
UNIVERSIDAD DE LAS AMÉRICAS
FACULTAD DE INGENIERÍA Y CIENCIAS APLICADAS
INGENIERÍA DE SOFTWARE
INTELIGENCIA ARTIFICIAL 1

CLASIFICACIÓN CON SVM
AUTOR: JOE CORDERO
"""

# Importación de librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga del dataset
# El dataset consiste en varias variables médicas predictoras (independientes) y una variable objetivo (dependiente), `Outcome`. 
# Las variables independientes incluyen el número de embarazos, el índice de masa corporal (BMI), nivel de insulina, edad, entre otros. 
# La variable `Outcome` indica si el paciente presenta diabetes (1) o no (0).
# Fuente: https://www.kaggle.com/code/mbalvi75/08-knn-diabetes-dataset/input
dataset = pd.read_csv("./IA 1/diabetes.csv")

# Selección de características y variable objetivo
# Seleccionamos las columnas 1 (Glucose) y 5 (BMI) como características independientes y la última columna (Outcome) como variable dependiente.
X = dataset.iloc[:, [1, 5]]  # Variables predictoras: Glucosa y BMI
y = dataset.iloc[:, -1]      # Variable objetivo: Diagnóstico de diabetes (1 = Sí, 0 = No)

# Imputacion de valores
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy="median")
imputer = imputer.fit(X)
X = imputer.transform(X)

# División de los datos en conjuntos de entrenamiento y prueba
# Utilizamos un 80% de los datos para entrenamiento y un 20% para prueba.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalización de las características
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Implementación del modelo SVM
# Utilizamos un kernel lineal.
from sklearn.svm import SVC
svma = SVC(kernel="linear", random_state=0)
svma.fit(X_train, y_train)  # Ajuste del modelo al conjunto de entrenamiento

# Predicción de resultados
y_pred = svma.predict(X_test)

# Evaluación del modelo mediante la matriz de confusión
from sklearn.metrics import confusion_matrix
confusion_matrix_result = confusion_matrix(y_test, y_pred)
# Visualización gráfica del modelo aplicado al conjunto de entrenamiento.
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, svma.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title("SVM (Conjunto de entrenamiento)")
plt.xlabel("Glucosa")
plt.ylabel("BMI")
plt.legend()
plt.show()

# Visualización gráfica del modelo aplicado al conjunto de prueba
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, svma.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title("SVM (Conjunto de prueba)")
plt.xlabel("Glucosa")
plt.ylabel("BMI")
plt.legend()
plt.show()
