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
n = int(sys.argv[3])
'''
filepath = "C:/Users/JIHYEON_KIM/Documents/workspace/rda/files/"
filename = "input3.csv"
n=3
'''
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


# In[5]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC


# In[7]:


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

feature_list.to_csv("./public/files/importance_score_" + filename + "_.csv", mode = "w", index = False)
feature_list.to_csv("./public/files/importance_score_result_" + filename + "_.csv", mode = "w",encoding='cp949',index = False)

# In[ ]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVR

X = data_na_remove.iloc[:,1:]
Y = data_na_remove.iloc[:,0]


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


feature_list.to_csv('./public/files/feature_selection_'+filename+'_.csv',na_rep='')
feature_list.to_csv('./public/files/feature_selection_result_'+filename+'_.csv',na_rep='',encoding='cp949')

#feature_list.to_csv('./feature_selection_'+filename+'_.csv',na_rep='')
# In[ ]:


#rf 실행
data = data_na_remove
y_data = data.iloc[:,0]
x_data = data.iloc[:,1:]
rf = RandomForestClassifier()
rf.fit(x_data, y_data)
features = x_data.columns.values


# In[ ]:


import plotly
import psutil
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
#scatter plot
trace = go.Scatter(
        y = rf.feature_importances_,
        x = features, 
        mode = 'markers',
        marker = dict(
                sizemode = 'diameter',
                sizeref = 1,
                size = 13, 
                color = rf.feature_importances_,
                colorscale = 'Portland',
                showscale = True),
        text =features)
data = [trace]

layout = go.Layout(
        autosize = True,
        title = 'Random Forest Feature Importance',
        hovermode = 'closest',
        xaxis = dict(
                ticklen = 5,
                showgrid = False,
                zeroline = False, 
                showline = False),
        yaxis = dict(
                title = 'Feature Importance',
                showgrid = False,
                zeroline = False,
                ticklen = 5, 
                gridwidth = 2),
        showlegend = False)
fig = go.Figure(data = data, layout = layout)
fig.write_image("./public/files/rf_feature_importance"+filename+"_.png")


# In[ ]:


#barplot
x, y = (list(x) for x in zip(*sorted(zip(rf.feature_importances_, features),reverse = False)))
trace2 = go.Bar(
        x = x,
        y = y,
        marker = dict(
                color = x,
                colorscale = 'Viridis',
                reversescale = True),
        name = 'Random Forest Feature importance',
        orientation = 'h'
        )
layout = dict(
        title = 'Barplot of Feature importances',
        width = 900,
        height = 2000,
        yaxis = dict(
                showgrid = False,
                showline = False, 
                showticklabels = True, 
                domain = [0,0.85],
        ))
fig1= go.Figure(data = [trace2])
fig1['layout'].update(layout)
fig1.write_image("./public/files/rf_feature_importance_barplot"+filename+"_.png")




import zipfile
os.chdir("./public/files/")
file_ls = ['importance_score_result_' + filename + '_.csv', 'rf_feature_importance'+filename+'_.png', 'rf_feature_importance_barplot'+filename+'_.png']
with zipfile.ZipFile('feature_selection_'+filename+'_.zip', 'w') as feature_zip:
    for i in file_ls:
        feature_zip.write(i)
    feature_zip.close()


