# -----------------------------------------------------------
# Tutorial de como utilizar o SVM com margem rígida para realizar
# classificações no Iris dataset.
#
# The Iris flower data set or Fisher's Iris data set is a multivariate data set
# introduced by the British statistician and biologist Ronald Fisher in his 1936
# paper The use of multiple measurements in taxonomic problems as an example of
# linear discriminant analysis.
#
# @author Douglas Moura
# -----------------------------------------------------------

import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# carrega dataset
iris = datasets.load_iris()

# matriz contendo apenas dois atributos
X = iris.data[:, (2, 3)] # petal length, petal width

# classes
y = iris.target

# exclui a classe virginica
setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# svm linear com margem rigida
svm_clf = SVC(kernel="linear", C=float("inf"))

svm_clf.fit(X, y)

# instancia de teste
x_test = np.array([[2.4, 0.8]])
h = svm_clf.predict(x_test)

print("Qual é a classe?", iris.target_names[h])

# plot
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plotando os limites de decisão

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# cria um grid
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)

YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm_clf.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
           
# plot support vectors
ax.scatter(svm_clf.support_vectors_[:, 0], svm_clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

ax.plot(x_test[:,0],x_test[:,1], "yo", c="red")

plt.show()