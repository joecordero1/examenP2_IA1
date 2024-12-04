# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
k debe ser impar
por default k es 5
toma 5 de la una y cinco de la otra
y mide la que está mas cercana


"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("./IA 1/diabetes.csv")

##vamos a saber que personas van a adquirir un vehículo en funcion del salario y edad
##ojo con confundirme con la multiple jaja

##Ahora debemos tomar una matriz que sea X
##aqui es la edad y el salario estimado
#Establecemos las variables independientes
X = dataset.iloc[:, [1,5]]
 
#Establecemos la variable dependiente
y = dataset.iloc[:, -1]

# Imputar valores faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy="median")
imputer = imputer.fit(X)
X = imputer.transform(X)

## Dividir el Dataset en conjunto de entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


## Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

## Ajuste del clasificador en el conjunto de entrenamiento
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2 )
knn.fit(X_train, y_train)
    
## Primer parámetro el número de vecinos, por default es 5
## 2do Parámetro tipo de distancia en P Si p=1 Manhattan, si p=2 Euclidiana
knn.fit(X_train, y_train)

## Predicción de los resultados con el conjunto de test
y_pred = knn.predict(X_test)

##matriz e confusion
from sklearn.metrics import confusion_matrix
conmet= confusion_matrix(y_test, y_pred)

### Visualizar el algotirmo de train graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2 , knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("K-NN (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
 
 
### Visualizar el algotirmo de test graficamente con los resultados
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2 , knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha = 0.75, cmap = ListedColormap(('red', 'green')))                     
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title("K-NN (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


### este metodo me sirve para poder saber la correlacion de la svariables y solo escoger aquellas mas importantes:
# Calcular la correlación de las variables de X con la variable objetivo Y
correlation = dataset.corr()

# Seleccionar las variables más correlacionadas con Y
correlation_with_outcome = correlation['Outcome'].drop('Outcome').abs()  # Correlación absoluta
top_features = correlation_with_outcome.sort_values(ascending=False).head(5)  # Top 2 variables

# Imprimir las variables más importantes
print("Las características más importantes son:")
print(top_features)