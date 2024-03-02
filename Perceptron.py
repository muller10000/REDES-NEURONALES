"""
El conjunto de datos flor Iris o conjunto de datos iris de Fisher es un conjunto de datos multivariante 
introducido por Ronald Fisher en su artículo de 1936 "The use of multiple measurements in taxonomic problems"
este conjunto de datos se convirtió en un caso de prueba típico en aprendizaje automático como en máquinas de vectores de soporte.
Las máquinas de vectores de soporte (del inglés support-vector machines, SVM) son un conjunto de algoritmos de aprendizaje 
supervisado desarrollados por Vladimir Vapnik y su equipo en los laboratorios de AT&T Bell.
Estos métodos están propiamente relacionados con problemas de clasificación y regresión.
Dado un conjunto de muestras podemos etiquetar las clases y formar una SVM para construir un modelo que prediga la clase de una nueva muestra.
En ese concepto de "separación óptima" es donde reside la característica fundamental de las SVM: este tipo de algoritmos buscan el hiperplano que tenga 
la máxima distancia (margen) con los puntos que estén más cerca de él mismo. Por eso también a veces se les conoce a las SVM como clasificadores 
de margen máximo. De esta forma, los puntos del vector que son etiquetados con una categoría estarán a un lado del hiperplano 
y los casos que se encuentren en la otra categoría estarán al otro lado.
"""
#lectura conjunto de datos
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#cargamos el conjunto de datos
iris_dataset = load_iris()

#visualizamos las etiquetas del conjunto de datos

#variación morfológica de la flor Iris de tres especies relacionadas
print(iris_dataset.target_names)
#el largo y ancho del sépalo y pétalo, en centímetros
print(iris_dataset.data)

# Leemos el conjunto de datos con la libreria Pandas

df = pd.DataFrame(np.c_[iris_dataset['data'], iris_dataset['target']], 
                  columns= iris_dataset['feature_names'] + ['target'])
print(df)

# Representacion grafica de dos dimensiones del conjunto de datos

fig = plt.figure(figsize=(10, 7))

plt.scatter(df["petal length (cm)"][df["target"] == 0], 
            df["petal width (cm)"][df["target"] == 0], c="b", label="setosa")

plt.scatter(df["petal length (cm)"][df["target"] == 1], 
            df["petal width (cm)"][df["target"] == 1], c="r", label="versicolor")

plt.xlabel("petal_length", fontsize=14)
plt.ylabel("petal_width", fontsize=14)
plt.legend(loc="lower right", fontsize=14)

plt.show() 

# Representacion grafica de tres dimensiones del conjunto de datos. se agrega una variable mas "ancho del sepalo"

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

ax.scatter3D(df["petal length (cm)"][df["target"] == 0], 
            df["petal width (cm)"][df["target"] == 0], 
            df["sepal width (cm)"][df["target"] == 0], c="b")

ax.scatter3D(df["petal length (cm)"][df["target"] == 1], 
            df["petal width (cm)"][df["target"] == 1], 
            df["sepal width (cm)"][df["target"] == 1], c="r")

ax.set_xlabel("petal_length")
ax.set_ylabel("petal_width")
ax.set_zlabel("sepal_width")


plt.show()

# Representacion grafica de tres dimensiones del conjunto de datos
# se agrega una nueva clase de datos pertenciente al target= 2, iris virginica.

fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")

ax.scatter3D(df["petal length (cm)"][df["target"] == 0], 
            df["petal width (cm)"][df["target"] == 0], 
            df["sepal width (cm)"][df["target"] == 0], c="b")

ax.scatter3D(df["petal length (cm)"][df["target"] == 1], 
            df["petal width (cm)"][df["target"] == 1], 
            df["sepal width (cm)"][df["target"] == 1], c="r")

ax.scatter3D(df["petal length (cm)"][df["target"] == 2], 
            df["petal width (cm)"][df["target"] == 2], 
            df["sepal width (cm)"][df["target"] == 2], c="y")

ax.set_xlabel("petal_length")
ax.set_ylabel("petal_width")
ax.set_zlabel("sepal_width")

plt.show()




