# -----------------------------------------------------------
# Tutorial de PCA sem o sklearn no iris dataset.
#
# The Iris flower data set or Fisher's Iris data set is a multivariate data set
# introduced by the British statistician and biologist Ronald Fisher in his 1936
# paper The use of multiple measurements in taxonomic problems as an example of
# linear discriminant analysis.
#
# @author Douglas Moura
# -----------------------------------------------------------

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import svd_flip

# carrega o dataset.
iris = datasets.load_iris()

# Matriz (150 x 4).
X = iris.data

# centraliza os dados na origem.
X_centered = (X - X.mean(axis=0))

# número de linhas - 1.
m = X_centered.shape[0] - 1

# calcula a matriz de covariância.
sigma = np.dot(X_centered.T, X_centered) / m

# calculo da matriz de covariância com numpy.
#cov = np.cov(X_centered.T, rowvar=False)

# decomposição em valores singulares.
U, s, V = np.linalg.svd(sigma)

# correção da saída do svd.
U, V = svd_flip(U, V)

# projeção dos dados utilizando os autovetores.
Z = np.dot(X_centered, V.T[:, 0:2])

# visualização em 2D.
plt.scatter(Z[:,0], Z[:,1], c=iris.target)
plt.show()