#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


# In[ ]:



filepath = sys.argv[1]
filename = sys.argv[2]
n = int(sys.argv[3])
'''
filepath = "C:/Users/JIHYEON_KIM/Documents/workspace/rda/files/"
filename = "input3.csv"
n=5
'''

data = pd.read_csv(filepath + "/" + filename, encoding='UTF-8')


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


X = data_na_remove.iloc[:,1:]
Y = data_na_remove.iloc[:,0]


# In[ ]:


#random forest
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
forest.fit(X, Y)
feature_list = pd.concat([pd.Series(X.columns), pd.Series(forest.feature_importances_)], axis=1)
feature_list.columns = ['features_name', 'importance']
feature_list_rf =feature_list.sort_values("importance", ascending =False)
rf_select = feature_list_rf.index[:(n)]


# In[ ]:


#L1 based Linear SVC
lsvc = LinearSVC(max_iter=1000000).fit(X, Y)
model = SelectFromModel(lsvc,prefit=True)
l1_select = model.get_support()


# In[ ]:


#L1 based logistic regression
lr = LogisticRegression(max_iter=10000)
selector = SelectFromModel(estimator=lr).fit(X, Y)
lr_select = selector.get_support()


# In[ ]:


#RFE
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select= n, step=1)
selector = selector.fit(X,Y)
rfe_select = selector.support_


# In[ ]:


rf_list = X.columns[rf_select]
l1_list = X.columns[l1_select]
lr_list = X.columns[lr_select]
rfe_list = X.columns[rfe_select]
feature_list = pd.concat([pd.Series(rf_list),pd.Series(l1_list), pd.Series(lr_list),pd.Series(rfe_list)], axis=1)
feature_list.columns = ['Random Forest','L1 based LinearSVC', 'L1 based Log Regression','RFE']


# In[ ]:


feature_list.to_csv('./public/files/feature_selection_'+filename+'_.csv',na_rep='',encoding='utf-8')
feature_list.to_csv('./public/files/feature_selection_result_'+filename+'_.csv',na_rep='',encoding='cp949')


