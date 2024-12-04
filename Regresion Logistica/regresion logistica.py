# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
REGRESION LOGISTICA

Aquí se usa para predecir variables categoricas
La palabra clave es CLASIFICAR

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:/Users/laboratorio/Downloads/Social_Network_Ads.csv")

##vamos a saber que personas van a adquirir un vehículo en funcion del salario y edad
##ojo con confundirme con la multiple jaja

##Ahora debemos tomar una matriz que sea X
##aqui es la edad y el salario estimado
X = dataset.iloc[:, [2,3]].values

## Saca todas las filas y la columna numero 3
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

##aqui escalo y les hago comparables, tanto a X train y a X test
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

##Aqui aplkico la regresion logistica, LogisticRegression es una CLASE, para que sea metodo debe comenxzar con minuscula
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

##aqui predecimos en base al train para poder validarlo con test
y_pred = classifier.predict(X_test)

##Ahora hacemos la MATRIZ DE CONFUSION
##nos sirve para saber si la prediccion fue buena
##0,0 es el NO difinitvo
##1,1 es el SI definitivo
##los valores como 0,1 y 1,0 son los errores
##0,1 es falso positivo
##1,0 es falso negativo (este es el mas preocupante)
##para calcular el procentaje de certeza sumamos en diagonal
##0,0 + 1,1   y el error es 0,1+1,0
from sklearn.metrics import confusion_matrix
conmet = confusion_matrix(y_test, y_pred)


##grafico
### Visualizar el algoritmo de train gráficamente con los resultados
##EN FUNCION DE TRAIN
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
plt.title("Classifier (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


##grafico
### Visualizar el algoritmo de train gráficamente con los resultados
##EN FUNCION DE TEST
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
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

##La matriz de confusion se puede comparar con los graficos
##en este caso puedo comparar la confusion con y_test y el grafico con y_test