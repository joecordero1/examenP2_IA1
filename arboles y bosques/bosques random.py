# -*- coding: utf-8 -*-
"""
Editor de Spyder

RANDOM FOREST

como funciona:
    selecciona un numero aleatorio de k pountso en un conjunto de entrenamiento
    se construye un arbol por cada k punto
    
    
    
    por defecto se toma 10 arboles
CUIDADO, este algoritmo tiene mucho sobreajuste, cuando las areas se hacen muy grandes hay mucho sobreajuste
    
para el examen debo decirle al profe cual de todos los modelos escogeria y justificar por qué
para el examen debo mostrarle al profe el modelo que estoy usando para el proyecto final
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

## Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


## Ajuste del clasificador en el conjunto de entrenamiento
from sklearn.ensemble import RandomForestClassifier
##aqui igual uso el criterio de entropia
##el numero de arbol nestimator, no afecta a la efectividad que se ve en la matriz de confusion, sino que solo clasifica mas finamente
rf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

#matriz de confusion
from sklearn.metrics import confusion_matrix
conmat = confusion_matrix(y_test, y_pred)

##grafrico de train
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Bosques aleatorios (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

##grafrico de test
from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, rf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title('Bosques aleatorios (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()