#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


#file 불러오기
file = "/home/data/projects/rda/workspace/rda/files/input3.csv"

data = pd.read_csv(file, header=0,encoding='utf-8')


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
#py.init_notebook_mode(connected = True)
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
fig.write_image("rf_feature_importance.png")


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
fig1.write_image("rf_feature_importance_barplot.png")

