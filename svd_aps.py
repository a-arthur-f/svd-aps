import numpy as np
from PIL import Image

img = np.array(Image.open('food.jpg'))  # Lê uma image e a transforma em uma matriz
original_shape = img.shape # Armazena a estrutura da matriz em uma variavel

# Cria uma matriz de duas dimensões com os valores da imagem
# possibilitando a decomposição SVD
img_reshaped = img.reshape(original_shape[0], original_shape[1] * 3)

# Armazena a quantidade de valores singulares que a mtriz possui
max_k = (original_shape[0] * original_shape[1] * original_shape[2]) // (original_shape[0] + 3*original_shape[1] + 1)
k = int(round(max_k * 0.8)) # Guarda 80% dos valores singulares

U, s, V = np.linalg.svd(img_reshaped, full_matrices=False) # Realiza a decomposição SVD

# Reconstroi a matriz de imagem usando apenas 80% dos valores
img_reconst = np.dot(U[:, :k], np.dot(np.diag(s[:k]), V[:k, :]))
img_reconst = img_reconst.reshape(original_shape).astype(np.ubyte)

Image.fromarray(img_reconst).show() # Mostra a imagem


#### A mesma imagem mostrada com 20% dos valores ####
k = int(round(max_k) * 0.2)

img_reconst = np.dot(U[:, :k], np.dot(np.diag(s[:k]), V[:k, :]))
img_reconst = img_reconst.reshape(original_shape).astype(np.ubyte)

Image.fromarray(img_reconst).show()