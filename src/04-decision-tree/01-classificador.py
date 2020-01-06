from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import os

iris = load_iris()

X = iris.data[:, 2:]

y = iris.target

tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X, y)

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