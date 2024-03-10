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

El proyecto es un testeo de la neurona Perceptron, encargada de procesar un conjunto de datos de entrenamiento y poder realizar predicciones 
sobre la variación morfológica de la flor Iris de tres especies relacionadas (Iris setosa, Iris virginica e Iris versicolor).
El codigo coloca en un data frame, la informacion sobre el ancho y el largo del sepalo y petalo (en centimetros) de cada flor.
De antemano se conoce la clasificacion de cada una, por lo que se podra averiguar la precision con la que funciona la neurona.

"""
#lectura conjunto de datos
from sklearn.datasets import load_iris
#libreria para importar la neurona
from sklearn.linear_model import Perceptron
#libreria para data frames
import pandas as pd
#libreria para arrays
import numpy as np
#libreria para graficos
import matplotlib.pyplot as plt
#libreria para 3d
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

#representaciones graficas de los datos

"""
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

"""

# Reducimos el conjunto de datos para entrenar el algoritmo y visualizar el resultado. nos quedamos solo con el ancho y el largo del petalo.
df_reduced = df[["petal length (cm)", "petal width (cm)", "target"]]
# nos quedamos solo con las primeros 2 tipos de flor, setosa y versicolor
df_reduced = df_reduced.loc[df_reduced["target"].isin([0, 1])]
print(df_reduced)

# Separamos las etiquetas de salida del resto de caracteristicas del conjunto de datos. Separamos las columnas de datos con la de target
X_df = df_reduced[["petal length (cm)", "petal width (cm)"]]
y_df = df_reduced["target"]
# el target solo queda con la clase 0 y 1 
print(y_df)

#se crea el perceptron y se lo entrena para ajustar parametros. con libreria sklearn
clf = Perceptron(max_iter=1000, random_state=40)
clf.fit(X_df, y_df)

# z(x) = x1*w1 + x2*w2 + b // se busca la funcion matematica que mejor separe los ejemplos de una clase y la otra

# x1*0.9 + x2*1.3 + (-3) // funcion optima. devuelve un valor continuo

# Parametros del modelo (ajusta el peso asociado a cada variable: w1 y w2)
print(clf.coef_)
# Terminio de interceptacion (ajusta el termino bias: b)
print(clf.intercept_)

# REPRESENTACION GRAFICA DEL LIMITE DE DECISION 

# me quedo con los valores de mi conjunto de datos de entrada
X = X_df.values

# agarro el valor minimo y maximo
mins= X.min(axis=0) - 0.1
maxs= X.max(axis=0) + 0.1 

# calculo valores intermedios entre el minimo y el maximo, concretamente 1000 valores
xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], 1000),
                     np.linspace(mins[1], maxs[1], 1000))

# metodo para predecir. en base a los 1000 valores que genere antes
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize=(10, 7))

# Represento la funcion matematica, "limite de decision"
plt.contourf(xx, yy, Z, cmap="Set3")
plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors='k')

# agrego sobre el anterior grafico los ejemplos del conjunto de datos, para comprobar si los has separado correctamente
plt.plot(X[:, 0][y_df==0], X[:, 1][y_df==0], 'bs', label="setosa")
plt.plot(X[:, 0][y_df==1], X[:, 1][y_df==1], 'go', label="vesicolor")

plt.xlabel("petal_length", fontsize=14)
plt.ylabel("petal_width", fontsize=14)
plt.legend(loc="lower right", fontsize=14)

plt.show()

# el limite negro que traza en la figura el PERCEPTRON, es un limite lineal, que dada una coordenada X e Y puede analizar el grafico y comprobar a que especie pertenece la flor
# a ese limite se le llama "limite de decision"

# entreno la neurona devuelta
clf = Perceptron(max_iter=1000, random_state=40)
clf.fit(X_df, y_df)

# esta vez sin pasarle el Y_df (etiqueta), y que almacene la prediccion en y_pred
y_pred = clf.predict(X_df)

from sklearn.metrics import accuracy_score

# comparo los resultados
print(accuracy_score(y_df, y_pred))

# como se puede ver en el grafico, ha separado las 2 clases con un 100% de precision







