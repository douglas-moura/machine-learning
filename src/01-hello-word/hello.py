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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# Load iris dataset
iris = datasets.load_iris()
print(list(iris.keys()))

# Sepal and petal length and width of 150 iris flowers
#print(iris.data)
#print(iris.feature_names)

# Classification: Iris-Setosa, Iris-Versicolor, and Iris-Virginica
#print(iris.target)
print(iris.target_names)
#print(iris.data.shape)

# Plot dataset
plt.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
plt.title('Iris Dataset')
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
#plt.show()

X, y = load_iris(return_X_y=True)


#print(X)

#print(y)

clf = svm.SVC()

clf.fit(X, y)

# KNN

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

#X_new = [[5, 3, 0.5, 1.5]]

h = knn.predict(X_test)

print("H - {} \n y - {}".format(iris.target_names[h], iris.target_names[y_test]))

mse = mean_squared_error(y_test, h)
print(mse)

cm = confusion_matrix(y_test, h)
#print(cm)

plot_confusion_matrix(knn, X_test, y_test, display_labels=iris.target_names,
                                 cmap=plt.cm.Blues)

plt.show()

#from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# img = ax.scatter(iris.data[:, 0], iris.data[:, 1], iris.data[:, 2], c=iris.data[:, 3], cmap=plt.hot())
# fig.colorbar(img)
# plt.show()