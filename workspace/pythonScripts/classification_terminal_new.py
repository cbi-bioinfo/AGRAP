#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from subprocess import call

import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# In[ ]:
#file 불러오기
filepath = sys.argv[1]
filename = sys.argv[2]
#filepath = "C:/Users/JIHYEON_KIM/Documents/workspace/rda/files/"
#filename = "input3.csv"

data = pd.read_csv(filepath + "/" + filename, encoding='UTF-8')


# In[ ]:


#사용자 지정 parameter
#svm

svm_kernel = sys.argv[3]
#knn
knn_neighbor = int(sys.argv[4])
#rf
rf_estimator = int(sys.argv[5])
rf_criteria = sys.argv[6]

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


#10cv

cv=KFold(n_splits=10)

svm_acc = []
rf_acc = []
lr_acc = []
knn_acc = []
nb_acc = []

for train_index ,test_index in cv.split(data_na_remove):

    train_cv= data_na_remove.iloc[train_index]       
    test_cv= data_na_remove.iloc[test_index]         
    
    train_X, test_X =train_cv.iloc[:,1:] , test_cv.iloc[:,1:]  #임의의 값.
    train_Y, test_Y =train_cv.iloc[:,0] , test_cv.iloc[:,0]  
    #svm
    clf = svm.SVC(kernel=svm_kernel)
    pred_svm = clf.fit(train_X, train_Y).predict(test_X) 
    acc_s = metrics.accuracy_score(test_Y, pred_svm)
    svm_acc.append(acc_s)
    #nb
    gnb = GaussianNB()
    pred_nb = gnb.fit(train_X, train_Y).predict(test_X) 
    acc_nb = metrics.accuracy_score(test_Y, pred_nb)
    nb_acc.append(acc_nb)
    #knn
    neigh = KNeighborsClassifier(n_neighbors= knn_neighbor)
    pred_knn = neigh.fit(train_X, train_Y).predict(test_X) 
    acc_knn = metrics.accuracy_score(test_Y, pred_knn)
    knn_acc.append(acc_knn)
    #lr
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pred_lr = pipe.fit(train_X, train_Y).predict(test_X) 
    acc_lr = metrics.accuracy_score(test_Y, pred_lr)
    lr_acc.append(acc_lr)
    #rf
    forest = RandomForestClassifier(n_estimators= rf_estimator, max_depth=2, criterion= rf_criteria)
    pred_rf = forest.fit(train_X, train_Y).predict(test_X) 
    acc_rf = metrics.accuracy_score(test_Y, pred_rf)
    rf_acc.append(acc_rf)   


# In[ ]:


#10cv dataframe 저장 
raw_data = {'svm': svm_acc,
            'knn': knn_acc,
            'nb': nb_acc,
            'lr': lr_acc,
            'rf':rf_acc}
 
data = pd.DataFrame(raw_data) 
data.to_csv('./public/files/10cv_acc_'+filename+'_.csv')


