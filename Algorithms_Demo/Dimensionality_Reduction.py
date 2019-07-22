import sklearn as sk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import FastICA
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.random_projection import SparseRandomProjection
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam

mnist = input_data.read_data_sets("MNIST_data/")
X_m = mnist.train.images
y = mnist.train.labels
pca = PCA(n_components=784, whiten = False, random_state = 2019)
X_pca = pca.fit_transform(X_m)
X_pca_reconst = pca.inverse_transform(X_pca)
plt.figure(figsize=(12,12))

plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], color='red', alpha=0.5,label='0')
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], color='blue', alpha=0.5,label='1')
plt.scatter(X_pca[y==2, 0], X_pca[y==2, 1], color='green', alpha=0.5,label='2')
plt.scatter(X_pca[y==3, 0], X_pca[y==3, 1], color='black', alpha=0.5,label='3')
plt.scatter(X_pca[y==4, 0], X_pca[y==4, 1], color='khaki', alpha=0.5,label='4')
plt.scatter(X_pca[y==5, 0], X_pca[y==5, 1], color='yellow', alpha=0.5,label='5')
plt.scatter(X_pca[y==6, 0], X_pca[y==6, 1], color='turquoise', alpha=0.5,label='6')
plt.scatter(X_pca[y==7, 0], X_pca[y==7, 1], color='pink', alpha=0.5,label='7')
plt.scatter(X_pca[y==8, 0], X_pca[y==8, 1], color='moccasin', alpha=0.5,label='8')
plt.scatter(X_pca[y==9, 0], X_pca[y==9, 1], color='olive', alpha=0.5,label='9')
plt.scatter(X_pca[y==10, 0], X_pca[y==10, 1], color='coral', alpha=0.5,label='10')
plt.title("PCA")
plt.ylabel('Les coordonnees de Y')
plt.xlabel('Les coordonnees de X')
plt.legend()
plt.show()
n_batches = 256
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_m, n_batches):
  inc_pca.partial_fit(X_batch)
X_ipca = inc_pca.transform(X_m)
X_ipca_reconst = inc_pca.inverse_transform(X_ipca)
kpca = KernelPCA(kernel="rbf",n_components=154, gamma=None, fit_inverse_transform=True, random_state = 2019, n_jobs=1)
kpca.fit(X_m[:10000,:])
X_kpca = kpca.transform(X_m)
X_kpca_reconst = kpca.inverse_transform(X_kpca)
sparsepca = SparsePCA(n_components=154, alpha=0.0001, random_state=2019, n_jobs=-1)
sparsepca.fit(X_m[:10000,:])
X_spacepca = sparsepca.transform(X_m)
SVD_ = TruncatedSVD(n_components=154,algorithm='randomized', random_state=2019, n_iter=5)
SVD_.fit(X_m[:10000,:])
X_svd = SVD_.transform(X_m)
X_svd_reconst = SVD_.inverse_transform(X_svd)
GRP = GaussianRandomProjection(n_components=154,eps = 0.5, random_state=2019)
GRP.fit(X_m[:10000,:])
X_grd = GRP.transform(X_m)
SRP = SparseRandomProjection(n_components=154,density = 'auto', eps = 0.5, random_state=2019, dense_output = False)
SRP.fit(X_m[:10000,:])
X_srp = SRP.transform(X_m)
mds = MDS(n_components=154, n_init=12, max_iter=1200, metric=True, n_jobs=4, random_state=2019)
X_mds = mds.fit_transform(X_m[:1000,:])
isomap = Isomap(n_components=154, n_jobs = 4, n_neighbors = 5)
isomap.fit(X_m[0:5000,:])
X_isomap = isomap.transform(X_m)
miniBatchDictLearning = MiniBatchDictionaryLearning(n_components=154,batch_size = 200,alpha = 1,n_iter = 25,  random_state=2019)
miniBatchDictLearning.fit(X_m[:10000,:])
X_batch = miniBatchDictLearning.fit_transform(X_m)
FastICA = FastICA(n_components=154, algorithm = 'parallel',whiten = True,max_iter = 100,  random_state=2019)
X_fica = FastICA.fit_transform(X_m)
X_fica_reconst = FastICA.inverse_transform(X_fica)
tsne = TSNE(n_components=2,learning_rate=300,perplexity = 30,early_exaggeration = 12,init = 'random',  random_state=2019)
X_tsne = tsne.fit_transform(X_m[:10000,:])
lle = LocallyLinearEmbedding(n_components=4, n_neighbors = 10,method = 'modified', n_jobs = 4,  random_state=2019)
lle.fit(X_m[:5000,:])
X_lle = lle.transform(X_m)
m = Sequential()
m.add(Dense(512,  activation='elu', input_shape=(784,)))
m.add(Dense(128,  activation='elu'))
m.add(Dense(2,    activation='linear', name="bottleneck"))
m.add(Dense(128,  activation='elu'))
m.add(Dense(512,  activation='elu'))
m.add(Dense(784,  activation='sigmoid'))
m.compile(loss='mean_squared_error', optimizer = Adam())
history = m.fit(X_m, X_m, batch_size=128, epochs=5, verbose=1)

encoder = Model(m.input, m.get_layer('bottleneck').output)
Zenc = encoder.predict(X_m)
Renc = m.predict(X_m)





