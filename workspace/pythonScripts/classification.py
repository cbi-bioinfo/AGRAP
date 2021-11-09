#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
import pandas as pd
import numpy as np
#file 불러오기
filepath = sys.argv[1]
filename = sys.argv[2]
#filepath = "C:/Users/JIHYEON_KIM/Documents/workspace/rda/files/"
#filename = "input3.csv"

data = pd.read_csv(filepath + "/" + filename, encoding='UTF-8')



# In[3]:


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


# In[4]:


data_na_remove


# In[11]:


from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC

cv=KFold(n_splits=10)

svm_acc = []
rf_acc = []
lr_acc = []
knn_acc = []
nb_acc = []


# In[12]:


for train_index ,test_index in cv.split(data_na_remove):

    train_cv= data_na_remove.iloc[train_index]       
    test_cv= data_na_remove.iloc[test_index]  
    
    train_X, test_X =train_cv.iloc[1:100,1:3] , test_cv.iloc[1:100,1:3]  #임의의 값.
    train_Y, test_Y =train_cv.iloc[1:100,0] , test_cv.iloc[1:100,0]  
    
    #svm
    clf = svm.SVC(kernel='linear')
    pred_svm = clf.fit(train_X, train_Y).predict(test_X) 
    acc_s = metrics.accuracy_score(test_Y, pred_svm)
    svm_acc.append(acc_s)
    #nb
    gnb = GaussianNB()
    pred_nb = gnb.fit(train_X, train_Y).predict(test_X) 
    acc_nb = metrics.accuracy_score(test_Y, pred_nb)
    nb_acc.append(acc_nb)
    #knn
    neigh = KNeighborsClassifier(n_neighbors=3)
    pred_knn = neigh.fit(train_X, train_Y).predict(test_X) 
    acc_knn = metrics.accuracy_score(test_Y, pred_knn)
    knn_acc.append(acc_knn)
    #lr
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pred_lr = pipe.fit(train_X, train_Y).predict(test_X) 
    acc_lr = metrics.accuracy_score(test_Y, pred_lr)
    lr_acc.append(acc_lr)
    #rf
    forest = RandomForestClassifier(max_depth=2, random_state=0)
    pred_rf = forest.fit(train_X, train_Y).predict(test_X) 
    acc_rf = metrics.accuracy_score(test_Y, pred_rf)
    rf_acc.append(acc_rf)    


# In[13]:


print("support vector machine accuracy",svm_acc)
print("naive bayes accuracy",nb_acc)
print("knn accuracy",knn_acc)
print("log regression accuracy",lr_acc) 
print("random forest accuracy",rf_acc)


# In[14]:


#10cv dataframe 저장 
raw_data = {'svm': svm_acc,
            'knn': knn_acc,
            'nb': nb_acc,
            'lr': lr_acc,
            'rf':rf_acc}
 
data = pd.DataFrame(raw_data) 
data.to_csv('./public/files/10cv_acc_'+filename+'_.csv')
'''
# In[15]:


#random forest feature importance
feature_list = data_na_remove.columns
for feature in feature_list :
	data = data_na_remove
	y_data = data.iloc[:,0]
	x_data = data.iloc[:,1:]

	svc = LinearSVC(random_state=0)
	rfe = RFE(estimator=svc, verbose = 1)

	rf_model = RandomForestClassifier()
	rf_model = rf_model.fit(x_data, y_data)

	feature_list = pd.concat([pd.Series(x_data.columns), pd.Series(rf_model.feature_importances_)], axis=1)
	feature_list.columns = ['features_name', 'importance']
	feature_list=feature_list.sort_values("importance", ascending =False)

#feature_list.to_csv("feature_importance_score_" + ".csv", mode = "w", index = False)
print(feature_list)


# In[16]:


#한글 깨짐 해결
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
import seaborn as sns
#%matplotlib inline   

#font 설정
fontlist = [(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]
ourfont = fontlist[1][0]
plt.rcParams["font.family"] = ourfont

# In[20]:


#yellowbrick 설치
import sys  

#get_ipython().system(u'{sys.executable} -m pip install --user graphviz')


# In[21]:


from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'decistion-tree.png', '-Gdpi=600'])

train, test = train_test_split(data_na_remove, test_size=0.33)
list = feature_list.index[0:3]
train_x, train_y = train.iloc[1:100,list], train.iloc[1:100,0]

model = RandomForestClassifier()
model.fit(train_x, train_y)
rf_estimator = model.estimators_[3]

c_name=[]
for i in data_na_remove['label']:
  c_name.append(str(i))


#random forest graph
export_graphviz(rf_estimator, out_file='tree.dot', 
                feature_names = train_x.columns,
                class_names = c_name,
                max_depth = 3, # 표현하고 싶은 최대 depth
                precision = 3, # 소수점 표기 자릿수
                filled = True, # class별 color 채우기
                rounded=False, # 박스의 모양을 둥글게
               )

#png 파일로 다시 내보내기
import pydot
import graphviz

(graph,) = pydot.graph_from_dot_file('tree.dot',encoding='utf-8')
#display(graphviz.Source(graph)) #보여주기
graph.write_png('./public/files/tree_'+filename+'_.png')
'''
