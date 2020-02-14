# -----------------------------------------------------------
# Tutorial de como utilizar o SVM com margem suave para realizar
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# carrega o iris dataset
iris = datasets.load_iris()

# matriz contendo apenas dois atributos
X = iris.data[:, (2, 3)] # petal length, petal width

# classes
y = iris.target

# exclui a classe setosa
virginica_or_versicolor = (y == 2) | (y == 1)
X = X[virginica_or_versicolor]
y = y[virginica_or_versicolor]

# svm linear com margem suave
svm_clf = SVC(kernel="linear", C=100) # teste com c = 1

# separa 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# treinamento
svm_clf.fit(X_train, y_train)

# teste
h = svm_clf.predict(X_test)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)

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

# limite de decisão e as margens
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# classificações incorretas
for i in range(0, len(y_test)):
    if h[i] != y_test[i]:
        ax.scatter(X_test[i, 0], X_test[i, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')

ax.plot(X_test[:,0],X_test[:,1], "yo", c="red")

# descomente para visualizar a matriz de confusão
#cm = confusion_matrix(y_test, h)
#plot_confusion_matrix(svm_clf, X_test, y_test, display_labels=iris.target_names[1:3],
#                                 cmap=plt.cm.Blues)

plt.show()