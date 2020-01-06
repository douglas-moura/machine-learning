# -----------------------------------------------------------
# Tutorial de PCA utilizando o sklearn no iris dataset.
#
# The Iris flower data set or Fisher's Iris data set is a multivariate data set
# introduced by the British statistician and biologist Ronald Fisher in his 1936
# paper The use of multiple measurements in taxonomic problems as an example of
# linear discriminant analysis.
#
# @author Douglas Moura
# -----------------------------------------------------------

from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt

# carrega dataset
iris = datasets.load_iris()

# Matriz (150 x 4).
X = iris.data

# pca com 2 componentes.
pca = PCA(n_components = 2)

# projeção dos dados.
Z = pca.fit_transform(X)

# Componentes principais.
print(pca.components_)

# taxa de variância.
print(pca.explained_variance_ratio_)

# visualização em 2D.
plt.scatter(Z[:, 0], Z[:, 1], c=iris.target)
plt.show()