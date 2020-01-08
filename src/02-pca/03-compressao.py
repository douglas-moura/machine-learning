from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np


# carrega dataset digits.
digits = load_digits()

X = digits.data

# pca com 2 componentes.
pca = PCA(n_components = 64)

# projeção dos dados.
Z = pca.fit_transform(X)

print(X.shape, Z.shape)

# Componentes principais.
print(pca.components_.shape)

# taxa de variância.
print(np.sum(pca.explained_variance_ratio_))
print(pca.explained_variance_)

digits_2 = pca.inverse_transform(Z)

# erro quadrático médio de projeção
erro = ((X - digits_2) ** 2).mean()
print(erro)

#plt.gray() 
plt.subplot(121)
plt.imshow(digits.images[0], cmap=plt.cm.gray_r) 
plt.subplot(122)
plt.imshow(digits_2[0].reshape(8, 8), cmap=plt.cm.gray_r)
plt.show()

#X_rec = np.dot(Z, V) # reconstrução

# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# print(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Número de componentes')
# plt.ylabel('Variancia preservada')
# plt.ylim(0,1)
# plt.show()