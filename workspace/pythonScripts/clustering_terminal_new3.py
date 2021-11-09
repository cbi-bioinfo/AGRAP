#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import pandas as pd
import numpy as np


# In[3]:


import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
import seaborn as sns


# In[4]:


#file 불러오기
file = sys.argv[1]
data = pd.read_csv(file, header=0,encoding='utf-8')


# In[ ]:


#사용자 지정 parameter
#kmeans 
k_clusters = sys.argv[2]
k_iter = sys.argv[3]
#dbscan
eps = sys.argv[4]
min_samples = sys.argv[5]
#hierarchy
h_clusters = sys.argv[6]


# In[ ]:


#모든 feature에 대해 결측치 갖는 샘플 제거
data_0 =data.dropna(axis=0,how='all')

#label 값이 결측치인 샘플 제거 
data_l =data.loc[data["label"].notnull(), :]

#50%이상이 결측치인 feature 삭제
data_f =data_l.dropna(axis=1,thresh=data_l.shape[0]/2)

#나머지는 각 label에 대해서 median imputation 수행
data_na_remove = data_f.fillna(data_f.mean())


# In[ ]:


data = data_na_remove.iloc[:,:]
X = data_na_remove.iloc[:,1:]
Y = data_na_remove.iloc[:,0] 


# In[ ]:


from sklearn.cluster import KMeans, DBSCAN ,AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
ari =[]
nmi =[]
silhouette =[]
sim = []
j_sim = []
pair_dis = []

#kmeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
predict_k = pd.DataFrame(kmeans.predict(X))
predict_k.columns=['predict_kmeans']
#concat
data_k = pd.concat([data,predict_k],axis=1)
#scores
ari.append(adjusted_rand_score(Y,kmeans.predict(X))) 
nmi.append(normalized_mutual_info_score(Y,kmeans.predict(X)))
silhouette.append(silhouette_score(X,kmeans.predict(X)))
sim.append(cosine_similarity(Y,kmeans.predict(X)))
j_sim.append(jaccard_similarity_score(Y,kmeans.predict(X)))
pair_dis.append(pairwise_distances(Y,kmeans.predict(X), metric='manhattan'))

#dbscan
dbscan = DBSCAN(eps=0.5,min_samples=5)
predict_db = pd.DataFrame(dbscan.fit_predict(X))
predict_db.columns=['predict_dbscan']
# concat
data_d = pd.concat([data_k,predict_db],axis=1)
#scores
ari.append(adjusted_rand_score(Y,dbscan.fit_predict(X)))
nmi.append(normalized_mutual_info_score(Y,dbscan.fit_predict(X))
silhouette.append(silhouette_score(X,dbscan.fit_predict(X)))
sim.append(cosine_similarity(Y,dbscan.fit_predict(X)))
j_sim.append(jaccard_similarity_score(Y,dbscan.fit_predict(X)))
pair_dis.append(pairwise_distances(Y,dbscan.predict(X), metric='manhattan'))

# hierarchy
hierarchy = AgglomerativeClustering(n_clusters=3)
predict_h = pd.DataFrame(hierarchy.fit_predict(X))
predict_h.columns=['predict_hierarchy']
#concat
data_h = pd.concat([data_d,predict_h],axis=1)
#scores
ari.append(adjusted_rand_score(Y,hierarchy.fit_predict(X)))
nmi.append(normalized_mutual_info_score(Y,hierarchy.fit_predict(X)))
silhouette.append(silhouette_score(X,hierarchy.fit_predict(X)))
sim.append(cosine_similarity(Y,hierarchy.fit_predict(X)))
j_sim.append(jaccard_similarity_score(Y,hierarchy.fit_predict(X)))
pair_dis.append(pairwise_distances(Y,hierarchy.fit_predict(X), metric='manhattan'))


#predict data save
data_h.to_csv('cluster_data.csv')

#clustering score save
score = pd.concat([pd.Series(ari),pd.Series(nmi), pd.Series(silhouette)], axis=1)
score.columns = ['ARI score','NMI score', 'Silhouette score']
score.index = ['Kmeans','DBScan','Hierarchy']
score.to_csv('clustering_score.csv')

#similarity score
sim_score = pd.concat([pd.Series(sim),pd.Series(j_sim),pd.Series(pair_dis)],axis=1)
score.columns = ['Cosine_similarity','Jaccard_similarity', 'Pairwise_distance']
score.index = ['Kmeans','DBScan','Hierarchy']
score.to_csv('clustering_similarity_score.csv')


# In[1]:


#silhouette graph
from yellowbrick.cluster import silhouette_visualizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouette_visualizer(KMeans(k_clusters, random_state=42), X, colors='yellowbrick')
plt.savefig('Silhouette_score.png')


# In[ ]:


#hierarchy dendrogram

from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()

labels = pd.DataFrame(Y)
labels.columns=['labels']
data = pd.concat([X,Y],axis=1)

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(data,method='complete')

# Plot the dendrogram, using varieties as labels
plt.figure(figsize=(40,20))
dendrogram(mergings,
           labels = labels.values,
           leaf_rotation=90,
           leaf_font_size=20,
)
plt.title('Dendrogram',fontsize=20)
plt.savefig('Dendrogram.png')

