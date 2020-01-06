from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


# carrega dataset digits.
digits = load_digits()

X = digits.data

# pca com 2 componentes.
pca = PCA(n_components = .99)

# projeção dos dados.
Z = pca.fit_transform(X)

print(X.shape, Z.shape)

# Componentes principais.
print(pca.components_.shape)

# taxa de variância.
#print(pca.explained_variance_ratio_)
#print(pca.explained_variance_)

digits_2 = pca.inverse_transform(Z)

#plt.gray() 
plt.subplot(330 + 1)
plt.imshow(digits.images[0], cmap=plt.cm.gray_r) 
plt.subplot(330 + 2)
plt.imshow(digits_2[0].reshape(8, 8), cmap=plt.cm.gray_r)
plt.show()

#X_rec = np.dot(Z, V) # reconstrução

#erro = np.absolute(X_rec - X_centered) # erro de reconstrução