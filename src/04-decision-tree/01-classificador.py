# -----------------------------------------------------------
# Tutorial introdutório sobre arvóres de decisão para realizar
# classificações no Iris dataset.
#
# The Iris flower data set or Fisher's Iris data set is a multivariate data set
# introduced by the British statistician and biologist Ronald Fisher in his 1936
# paper The use of multiple measurements in taxonomic problems as an example of
# linear discriminant analysis.
#
# @author Douglas Moura
# -----------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import os

# carrega o iris dataset
iris = load_iris()

# matriz contendo apenas dois atributos
X = iris.data[:, (2, 3)] # petal length, petal width

# classes
y = iris.target

# criação da arvore
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X, y)

# salva as regras de classifição
export_graphviz(
    tree,
    out_file=os.path.join("figures", "iris_tree.dot"),
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

# rode para converter para png
#dot -Tpng figures/iris_tree.dot -o figures/iris_tree.png  