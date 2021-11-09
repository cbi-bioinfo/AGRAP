#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import os
import sys

#file 불러오기
#filepath = sys.argv[1]
#filename = sys.argv[2]
filepath = "/home/data/projects/rda/workspace/rda/files/"
filename = "input3.csv"

data = pd.read_csv(filepath + "/" + filename, encoding='UTF-8')

# In[14]:


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


# In[16]:


data_na_remove


# In[17]:


print(data_na_remove.shape)
data = data_na_remove.iloc[:100,:5]
X = data_na_remove.iloc[:100,1:5]
Y = data_na_remove.iloc[:100,0] #임의의 
data_na_remove["label"].unique()


# In[18]:


from sklearn.cluster import KMeans, DBSCAN ,AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
ari =[]
nmi =[]
silhouette =[]
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

#dbscan
dbscan = DBSCAN(eps=0.5,min_samples=5)
predict_db = pd.DataFrame(dbscan.fit_predict(X))
predict_db.columns=['predict_dbscan']
# concat
data_d = pd.concat([data_k,predict_db],axis=1)
#scores
ari.append(adjusted_rand_score(Y,dbscan.fit_predict(X)))
nmi.append(normalized_mutual_info_score(Y,dbscan.fit_predict(X)))
silhouette.append(silhouette_score(X,dbscan.fit_predict(X)))


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

#data save
data_h.to_csv('./cluster_data_' + filename + '_.csv', mode = "w",encoding='cp949')
#clustering score save
score = pd.concat([pd.Series(ari),pd.Series(nmi), pd.Series(silhouette)], axis=1)
score.columns = ['ARI score','NMI score', 'Silhouette score']
score.index = ['Kmeans','DBScan','Hierarchy']
score.to_csv('./clustering_score_'+filename+'_.csv')

# In[19]:

'''
#yellowbrick 설치
import sys  
#get_ipython().system(u'{sys.executable} -m pip install --user yellowbrick')


# In[28]:


#silhouette graph
from yellowbrick.cluster import silhouette_visualizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouette_visualizer(KMeans(3, random_state=42), X, colors='yellowbrick')
plt.savefig('./public/files/Silhouette_score_' + filename + '_.png')


# In[53]:


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
file_ls = ['cluster_data_' + filename + '_.csv','Silhouette_score_' + filename + '_.png','Dendrogram_' + filename + '_.png', 'clustering_score_'+filename+'_.csv']
with zipfile.ZipFile('clustering_'+filename+'_.zip', 'w') as cluster_zip:
    for i in file_ls:
        cluster_zip.write(i)
    cluster_zip.close()

# In[ ]:
'''



