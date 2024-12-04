# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 23:53:52 2024

@author: joema
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Cargar datos
# Este dataset contiene información relevanet acerca de un estudio cardiovascular
# de habitantes de la ciudad Framingham, Massachusetts.
# Fuente: https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression?resource=download
dataset = pd.read_csv("./IA 1/framingham.csv")

# Seleccionar variables más importantes
X = dataset.iloc[:, [1, 10]].values  # 'age' y 'sysBP'
y = dataset.iloc[:, 15].values

# Imputar valores faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="median")
imputer = imputer.fit(X)
X = imputer.transform(X)

# Dividir datos en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Escalar datos
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Entrenar modelo
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predecir resultados
y_pred = classifier.predict(X_test)

# Matriz de confusión
from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)

# Graficar con solo las dos variables más importantes ('age' y 'sysBP')
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Classifier (test set)")
plt.xlabel("Edad")
plt.ylabel("Systolic blood pressure")
plt.legend()
plt.show()

# Graficar con solo las dos variables más importantes ('age' y 'sysBP')
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("Classifier (test set)")
plt.xlabel("Edad")
plt.ylabel("Systolic blood pressure")
plt.legend()
plt.show()
