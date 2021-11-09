#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import pandas as pd
import numpy as np


# In[3]:


# In[4]:



#file 불러오기
filepath = sys.argv[1]
filename = sys.argv[2]
#filepath = "C:/Users/JIHYEON_KIM/Documents/workspace/rda/files/"
#filename = "input3.csv"

data = pd.read_csv(filepath + "/" + filename, encoding='UTF-8')

# In[ ]:


#사용자 지정 parameter
#kmeans

k_clusters = int(sys.argv[3])
k_iter = int(sys.argv[4])
#dbscan
eps = float(sys.argv[5])
min_samples = int(sys.argv[6])
#hierarchy
h_clusters = int(sys.argv[7])

# In[ ]:


#모든 feature에 대해 결측치 갖는 샘플 제거
data_0 =data.dropna(axis=0,how='all')
#label 값이 결측치인 샘플 제거
data_l =data.loc[data["label"].notnull(), :]
#50%이상이 결측치인 feature 삭제
data_f =data_l.dropna(axis=1,thresh=data_l.shape[0]/2)
#나머지는 각 label에 대해서 median imputation 수행
data_na_remove = data_f.fillna(data_f.mean())

# In[17]:


data = data_na_remove.iloc[:100,:5]
X = data_na_remove.iloc[:100,1:5]
Y = data_na_remove.iloc[:100,0] #임의의
data_na_remove["label"].unique()


# In[ ]:


from sklearn.cluster import KMeans, DBSCAN ,AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import pairwise_distances
ari =[]
nmi =[]
silhouette =[]
sim = []
j_sim = []
pair_dis = []
#kmeans
kmeans = KMeans(n_clusters= k_clusters,max_iter=k_iter).fit(X)
predict_k = pd.DataFrame(kmeans.predict(X))
predict_k.columns=['predict_kmeans']
#concat
data_k = pd.concat([data,predict_k],axis=1)

#scores
ari.append(adjusted_rand_score(Y,kmeans.predict(X)))
nmi.append(normalized_mutual_info_score(Y,kmeans.predict(X)))
silhouette.append(silhouette_score(X,kmeans.predict(X)))
sim.append(cosine_similarity(np.array(Y).reshape(-1, 1),np.array(kmeans.predict(X)).reshape(-1, 1)))
j_sim.append(jaccard_similarity_score(Y,kmeans.predict(X)))
pair_dis.append(pairwise_distances(np.array(Y).reshape(-1, 1),np.array(kmeans.predict(X)).reshape(-1, 1), metric='manhattan'))

#dbscan
dbscan = DBSCAN(eps= eps,min_samples= min_samples)
predict_db = pd.DataFrame(dbscan.fit_predict(X))
predict_db.columns=['predict_dbscan']
# concat
data_d = pd.concat([data_k,predict_db],axis=1)
#scores
ari.append(adjusted_rand_score(Y,dbscan.fit_predict(X)))
nmi.append(normalized_mutual_info_score(Y,dbscan.fit_predict(X)))
silhouette.append(silhouette_score(X,dbscan.fit_predict(X)))
sim.append(cosine_similarity(np.array(Y).reshape(-1, 1),np.array(dbscan.fit_predict(X)).reshape(-1, 1)))
j_sim.append(jaccard_similarity_score(Y,dbscan.fit_predict(X)))
pair_dis.append(pairwise_distances(np.array(Y).reshape(-1, 1),np.array(dbscan.fit_predict(X)).reshape(-1, 1), metric='manhattan'))

# hierarchy
hierarchy = AgglomerativeClustering(n_clusters= h_clusters)
predict_h = pd.DataFrame(hierarchy.fit_predict(X))
predict_h.columns=['predict_hierarchy']
#concat
data_h = pd.concat([data_d,predict_h],axis=1)
#scores
ari.append(adjusted_rand_score(Y,hierarchy.fit_predict(X)))
nmi.append(normalized_mutual_info_score(Y,hierarchy.fit_predict(X)))
silhouette.append(silhouette_score(X,hierarchy.fit_predict(X)))
sim.append(cosine_similarity(np.array(Y).reshape(-1, 1),np.array(hierarchy.fit_predict(X)).reshape(-1, 1)))
j_sim.append(jaccard_similarity_score(Y,hierarchy.fit_predict(X)))
pair_dis.append(pairwise_distances(np.array(Y).reshape(-1, 1),np.array(hierarchy.fit_predict(X)).reshape(-1, 1), metric='manhattan'))

#data save
data_h.to_csv('./public/files/cluster_data2_' + filename + '_.csv')
#data_h.to_csv('./cluster_data2_' + filename + '_.csv', mode = "w",encoding='cp949')
#clustering score save
score = pd.concat([pd.Series(ari),pd.Series(nmi), pd.Series(silhouette)], axis=1)
score.columns = ['ARI score','NMI score', 'Silhouette score']
score.index = ['Kmeans','DBScan','Hierarchy']
score.to_csv('./public/files/clustering_score_'+filename+'_.csv')
#similarity score
sim_score = pd.concat([pd.Series(sim),pd.Series(j_sim),pd.Series(pair_dis)],axis=1)
score.columns = ['Cosine_similarity','Jaccard_similarity', 'Pairwise_distance']
score.index = ['Kmeans','DBScan','Hierarchy']
score.to_csv('./public/files/clustering_similarity_score_'+filename+'_.csv')
# In[1]:

#data
table = data_na_remove.iloc[:,1:]
target = data_na_remove.iloc[:,0]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
result = scaler.fit_transform(table)
data_scaled = pd.DataFrame(result, columns=table.columns)

from sklearn.decomposition import PCA

pca = PCA(n_components=k_clusters)
result = pca.fit_transform(data_scaled)
result = pd.DataFrame(result, columns=["x", "y"])

target = pd.DataFrame(target, columns=['label'])
merged = pd.concat([result, target], axis=1)

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns

# %matplotlib inline

# font 정의
'''mlp.rcParams['font.size'] = 20
mlp.rcParams['font.family'] = 'Nanum Gothic'
'''
import matplotlib.font_manager as fm
fontlist = [(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]
ourfont = fontlist[1][0]
plt.rcParams["font.family"] = ourfont

# 시각화
plt.figure(figsize=(16, 9))
sns.set_palette(sns.color_palette("muted"))

sns.scatterplot(merged['x'],
                     merged['y'],
                     hue=merged['label'],
                     s=100,
                     palette=sns.color_palette('muted', n_colors=5),
                    )
plt.title('data plot')
plt.savefig('./public/files/pca_'+filename+'_.png')
#silhouette graph
from yellowbrick.cluster import silhouette_visualizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouette_visualizer(KMeans(k_clusters, random_state=42), X, colors='yellowbrick')
plt.savefig('./public/files/Silhouette_score_' + filename + '_.png')


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
plt.savefig('./public/files/Dendrogram_' + filename + '_.png')



import zipfile
os.chdir("./public/files/")
file_ls = ['cluster_data_' + filename + '_.csv','clustering_similarity_score_'+filename+'_.csv','Silhouette_score_' + filename + '_.png','Dendrogram_' + filename + '_.png', 'clustering_score_'+filename+'_.csv', 'pca_'+filename+'_.png']
with zipfile.ZipFile('clustering_'+filename+'_.zip', 'w') as cluster_zip:
    for i in file_ls:
        cluster_zip.write(i)
    cluster_zip.close()
"""     
file_ls = ['cluster_data_' + filename + '_.csv','clustering_similarity_score_'+filename+'_.csv','Silhouette_score_' + filename + '_.png','Dendrogram_' + filename + '_.png', 'clustering_score_'+filename+'_.csv']
with zipfile.ZipFile('clustering_'+filename+'_.zip', 'w') as cluster_zip:
    for i in file_ls:
        cluster_zip.write(i)
    cluster_zip.close() """