import matplotlib.pyplot as plt
import numpy as np

from sklearn import cluster, datasets
from sklearn.feature_extraction.image import grid_to_graph

# Carrega o conjunto de dados de dígitos
digits = datasets.load_digits()
images = digits.images  # Obtém as imagens dos dígitos

# Reformata as imagens em um vetor unidimensional para análise
X = np.reshape(images, (len(images), -1))
# Cria uma conectividade baseada na grade da primeira imagem
connectivity = grid_to_graph(*images[0].shape)

# Aplica o algoritmo de Feature Agglomeration com 32 clusters
agglo = cluster.FeatureAgglomeration(connectivity=connectivity, n_clusters=32)
agglo.fit(X)

# Transforma os dados originais para um espaço reduzido
X_reduced = agglo.transform(X)

# Reconstrói os dados originais a partir dos dados reduzidos
X_restored = agglo.inverse_transform(X_reduced)
images_restored = np.reshape(X_restored, images.shape)

# Configura a plotagem das imagens
plt.figure(1, figsize=(4, 3.5))  # Cria uma figura com tamanho específico
plt.clf()  # Limpa a figura atual, se existir
# Ajusta os espaços entre subplots para melhor visualização
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.91)

# Plota as imagens originais e reconstruídas
for i in range(4):
   # Subplot para a imagem original
   plt.subplot(3, 4, i + 1)
   plt.imshow(images[i], cmap=plt.cm.gray, vmax=16, interpolation="nearest")
   plt.xticks(())  # Remove os ticks do eixo x
   plt.yticks(())  # Remove os ticks do eixo y
   if i == 1:
       plt.title("Original data")  # Adiciona título para a segunda imagem original
       
   # Subplot para a imagem reconstruída após aglomeração de características
   plt.subplot(3, 4, 4 + i + 1)
   plt.imshow(images_restored[i], cmap=plt.cm.gray, vmax=16, interpolation="nearest")
   plt.xticks(())  # Remove os ticks do eixo x
   plt.yticks(())  # Remove os ticks do eixo y
   if i == 1:
       plt.title("Agglomerated data")  # Adiciona título para a segunda imagem reconstruída
       
# Subplot para plotar os rótulos dos clusters
plt.subplot(3, 4, 10)
plt.imshow(
   np.reshape(agglo.labels_, images[0].shape),
   interpolation="nearest",
   cmap=plt.cm.nipy_spectral,
)

plt.xticks(())  # Remove os ticks do eixo x
plt.yticks(())  # Remove os ticks do eixo y
plt.title("Labels")  # Adiciona título para o subplot dos rótulos dos clusters
plt.show()  # Exibe a figura final com todos os subplots