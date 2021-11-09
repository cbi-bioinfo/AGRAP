# -*- coding: utf-8 -*-

"""Corrrelation_terminal.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1H_Xv2-Mg-vCRn04X1aIHKgdWDg3E9gx5
"""

import os
import sys
import pandas as pd
import numpy as np

#file 불러오기
#file = sys.argv[1]
#data = pd.read_csv(file, header=0,encoding='utf-8')
#지현
filepath = sys.argv[1]
filename = sys.argv[2]
#filepath = "/home/data/projects/rda/workspace/rda/files/"
#filename = "input3.csv"

data = pd.read_csv(filepath + "/" + filename, encoding='UTF-8')
#모든 feature에 대해 결측치 갖는 샘플 제거
data_0 =data.dropna(axis=0,how='all')

#label 값이 결측치인 샘플 제거 
data_l =data.loc[data["label"].notnull(), :]

#50%이상이 결측치인 feature 삭제
data_f =data_l.dropna(axis=1,thresh=data_l.shape[0]/2)

#나머지는 각 label에 대해서 median imputation 수행
data_na_remove = data_f.fillna(data_f.mean())

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline   
#지현
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import seaborn as sns

plt.rc('font', family='NanumBarunGothic') 
plt.figure(figsize=(15,15))
sns.heatmap(data = data_na_remove.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='RdYlBu_r')
plt.savefig('./public/files/corr_heatmap_'+filename+'_.png')
#plt.savefig('./corr_heatmap_'+filename+'_.png')

