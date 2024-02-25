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

from sklearn.datasets import load_iris

iris_dataset = load_iris()

iris_dataset.target_names
