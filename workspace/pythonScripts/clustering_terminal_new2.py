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
#filepath = sys.argv[1]
#filename = sys.argv[2]
filepath = "/home/data/projects/rda/workspace/rda/files/"
filename = "input3.csv"

data = pd.read_csv(filepath + "/" + filename, encoding='UTF-8')

# In[ ]:


#사용자 지정 parameter
#kmeans 
'''
k_clusters = int(sys.argv[3])
k_iter = int(sys.argv[4])
#dbscan
eps = float(sys.argv[5])
min_samples = int(sys.argv[6])
#hierarchy
h_clusters = int(sys.argv[7])
'''
# In[ ]:
k_clusters = 5
k_iter = 300
#dbscan
eps = 0.5
min_samples =3
#hierarchy
h_clusters = 3


#모든 feature에 대해 결측치 갖는 샘플 제거
data_0 =data.dropna(axis=0,how='all')
print(data_0.shape)
#label 값이 결측치인 샘플 제거 
data_l =data.loc[data["label"].notnull(), :]
print(data_l.shape)
#50%이상이 결측치인 feature 삭제
data_f =data_l.dropna(axis=1,thresh=data_l.shape[0]/2)
print(data_f.shape)
#나머지는 각 label에 대해서 median imputation 수행
data_na_remove = data_f.fillna(data_f.mean())
print(data_na_remove.shape)


data_na_remove


# In[17]:


print(data_na_remove.shape)
data = data_na_remove.iloc[:100,:5]
X = data_na_remove.iloc[:100,1:5]
Y = data_na_remove.iloc[:100,0] #임의의 
data_na_remove["label"].unique()


# In[ ]:


from sklearn.cluster import KMeans, DBSCAN ,AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
ari =[]
nmi =[]
silhouette =[]
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

#data save
#data_h.to_csv('./public/files/cluster_data2_' + filename + '_.csv')
#data_h.to_csv('./cluster_data2_' + filename + '_.csv', mode = "w",encoding='cp949')
#clustering score save
score = pd.concat([pd.Series(ari),pd.Series(nmi), pd.Series(silhouette)], axis=1)
score.columns = ['ARI score','NMI score', 'Silhouette score']
score.index = ['Kmeans','DBScan','Hierarchy']
#score.to_csv('./public/files/clustering_score_'+filename+'_.csv')

# In[1]:


#silhouette graph
from yellowbrick.cluster import silhouette_visualizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouette_visualizer(KMeans(k_clusters, random_state=42), X, colors='yellowbrick')
plt.savefig('./Silhouette_score_' + filename + '_.png')

