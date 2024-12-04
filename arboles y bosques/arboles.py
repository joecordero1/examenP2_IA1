# -*- coding: utf-8 -*-
"""
Editor de Spyder

ARBOL DE DECISION

leer gini y entropia, ya que es pregunta de examen
la regresion es numerica
la clasificacon es categorica
aqui no hay distancias por lo que no se debe escalar, aunque si podría hacerlo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:/Users/laboratorio/Downloads/Social_Network_Ads.csv")

##vamos a saber que personas van a adquirir un vehículo en funcion del salario y edad
##Ahora debemos tomar una matriz que sea X
##aqui es la edad y el salario estimado
#Establecemos las variables independientes
#en arboles de decision se debe poner el .values, esto sirve para hacerle TUPLA
X = dataset.iloc[:, [2,3]].values

y = dataset.iloc[:,4].values    

## Dividir el Dataset en conjunto de entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Ajuste del clasificador en el conjunto de entrenamiento
from sklearn.tree import DecisionTreeClassifier
#el criterio para cladificar por default es Inidice Gini.
##aqui uso la entropia
dt = DecisionTreeClassifier(criterion="entropy", random_state=0)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

#matriz de confusion
from sklearn.metrics import confusion_matrix
conmat = confusion_matrix(y_test, y_pred)

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=1),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=500))

plt.contourf(X1, X2, dt.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Árbol de decisión (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()