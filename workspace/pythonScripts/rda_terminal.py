# -*- coding: utf-8 -*-
#"""RDA_terminal.ipynb

#Automatically generated by Colaboratory.

#Original file is located at
#    https://colab.research.google.com/drive/1HKE5wkN5lLqJGxZSxwhIeglY0ldcazp4
#"""

import os
#print(os.getcwd())
import zipfile
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

#file 불러오기
#모든 feature에 대해 결측치 갖는 샘플 제거
#지현


#filepath="/home/data/projects/rda/workspace/rda/files"
#filename="input3.csv"
filepath = sys.argv[1]
filename = sys.argv[2]

data = pd.read_csv(filepath + "/" + filename, encoding='UTF-8')
'''
printResult = str({"filePath": filepath, "resultFileName": "10cv_acc_"+filename+"_.csv"})
print(printResult.encode('utf-8'))
resultFile = filepath + "/" + "10cv_acc_" + filename+"_.csv" 
'''
#data = pd.read_csv(file, header=0,encoding='utf-8')
data_0 =data.dropna(axis=0,how='all')


#label 값이 결측치인 샘플 제거 
data_l =data.loc[data["label"].notnull(), :]

#50%이상이 결측치인 feature 삭제
data_f =data_l.dropna(axis=1,thresh=data_l.shape[0]/2)

#나머지는 각 label에 대해서 median imputation 수행
data_na_remove = data_f.fillna(data_f.mean())

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
    
    train_X, test_X =train_cv.iloc[:,1:] , test_cv.iloc[:,1:]  
    train_Y, test_Y =train_cv.iloc[:,0] , test_cv.iloc[:,0]  
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
    lr = LogisticRegression(random_state=0)
    pred_lr = lr.fit(train_X, train_Y).predict(test_X) 
    acc_lr = metrics.accuracy_score(test_Y, pred_lr)
    lr_acc.append(acc_lr)
    #rf
    forest = RandomForestClassifier(max_depth=2, random_state=0)
    pred_rf = forest.fit(train_X, train_Y).predict(test_X) 
    acc_rf = metrics.accuracy_score(test_Y, pred_rf)
    rf_acc.append(acc_rf)

#10cv dataframe 저장 
raw_data = {'SVM': svm_acc,
            'KNN': knn_acc,
            'NB': nb_acc,
            'LR': lr_acc,
            'RF':rf_acc}
 
data = pd.DataFrame(raw_data)
#지현
#resultFile=data.to_csv(resultFile)
#도경
#f = open(resultFile,'w')
#f.write(data) 

data.to_csv('./public/files/10cv_acc_'+filename+'_.csv')

#data.to_csv('./10cv_acc_'+filename+'_.csv')
#f.close()
# Commented out IPython magic to ensure Python compatibility.
#구글 코랩 한글 깨지는 현상 해결

import matplotlib as mpl
import matplotlib.pyplot as plt
 
# %config InlineBackend.figure_format = 'retina'
 
#!apt -qq -y install fonts-nanum
 
import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager._rebuild()

#random forest
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from subprocess import call
#call(['dot', '-Tpng', 'tree.dot', '-o', 'decistion-tree.png', '-Gdpi=600'],shell=True)

train, test = train_test_split(data_na_remove, test_size=0.33)
train_x, train_y = train.iloc[:,1:], train.iloc[:,0]

model = RandomForestClassifier()
model.fit(train_x, train_y)
rf_estimator = model.estimators_[3] #임의의 model 

c_name=[]
for i in data_na_remove['label']:
  c_name.append(str(i))
"""
#random forest graph
export_graphviz(rf_estimator, out_file='tree.dot', 
                feature_names = train_x.columns,
                class_names = c_name,
                max_depth = 3, # 표현하고 싶은 최대 depth
                precision = 3, # 소수점 표기 자릿수
                filled = True, # class별 color 채우기
                rounded=True, # 박스의 모양을 둥글게
               )

#png 파일로 다시 내보내기
import pydot
(graph,) = pydot.graph_from_dot_file('tree.dot',encoding='utf-8')
#display(graphviz.Source(graph)) #보여주기
graph.write_png('./public/files/tree_'+filename+'_.png')
"""
#random forest feature importance
feature_list = train_x.columns
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

os.chdir("./public/files/")
file_ls2 = ['10cv_acc_'+filename+'_.csv', 'importance_score_result_'+filename+'_.csv']
with zipfile.ZipFile('classification_'+filename+'_.zip', 'w') as rda_zip:
    for i in file_ls2:
        rda_zip.write(i)
    rda_zip.close()
