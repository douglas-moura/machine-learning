# -----------------------------------------------------------
# Tutorial inicial de como carregar um dataset e utilizar um algoritmo
# de Machine learning para classificação.
#
# The Iris flower data set or Fisher's Iris data set is a multivariate data set
# introduced by the British statistician and biologist Ronald Fisher in his 1936
# paper The use of multiple measurements in taxonomic problems as an example of
# linear discriminant analysis.
#
# @author Douglas Moura
# -----------------------------------------------------------

from sklearn import datasets
from sklearn.datasets import load_iris #
from sklearn import svm #
import matplotlib.pyplot as plt

# carrega o iris dataset
iris = datasets.load_iris()
print(list(iris.keys()))

# conhecendo o dataset
print(iris.data.shape) # dimensão da matriz

# atributos
print(iris.feature_names) # Sepal and petal length and width of 150 iris flowers

# dados
print(iris.data)


# classes: Iris-Setosa, Iris-Versicolor, and Iris-Virginica
print(iris.target) # ou print(iris.target_names)

# plotando o dataset com dois atributos
plt.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
plt.title('Iris Dataset')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.show()