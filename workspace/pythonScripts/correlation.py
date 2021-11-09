#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import pandas as pd
import numpy as np


# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
import seaborn as sns
#%matplotlib inline  


# In[ ]:


#file 불러오기
#file 불러오기
filepath = sys.argv[1]
filename = sys.argv[2]
#filepath = "C:/Users/JIHYEON_KIM/Documents/workspace/rda/files/"
#filename = "input3.csv"

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


#spearman 
from scipy import stats
sarray = stats.spearmanr(data_na_remove)[0] # 0은 correaltion, 1은 pvalue
smatrix = pd.DataFrame(sarray)
#corr matrix 이름 지정
smatrix.columns = data_na_remove.columns
smatrix.index = data_na_remove.columns


# In[ ]:


#font 설정
fontlist = [(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]
ourfont = fontlist[1][0]
plt.rcParams["font.family"] = ourfont


# In[ ]:


#heatmap pearson
plt.rc('font', family= ourfont)
plt.figure(figsize=(15,15))
sns.heatmap(data = data_na_remove.corr(), annot=True, 
fmt = '.2f', linewidths=.5, cmap='RdYlBu_r')
plt.title('Pearson Correlation Heatmap', fontsize=20)
plt.savefig('./public/files/pearson_corr_heatmap_'+filename+'_.png')


# In[ ]:


#heatmap spearman
plt.rc('font', family= ourfont) 
plt.figure(figsize=(15,15))
sns.heatmap(data = smatrix, annot=True, 
fmt = '.2f', linewidths=.5, cmap='RdYlGn_r')
plt.title('Spearman Correlation Heatmap', fontsize=20)
plt.savefig('./public/files/spearman_corr_heatmap_'+filename+'_.png')


# In[ ]:


#삼각형 pearson correlation heatmap
df = data_na_remove.corr()
# 그림 사이즈 지정
fig, ax = plt.subplots( figsize=(15,15) )

# 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
mask = np.zeros_like(df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 히트맵을 그린다
plt.rc('font', family= ourfont) 
sns.heatmap(df, 
            cmap = 'RdYlBu_r', 
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.title('Pearson triangle Correlation Heatmap', fontsize=20)
plt.savefig('./public/files/pearson_corr_tri_heatmap_'+filename+'_.png')


# In[ ]:


#삼각형 spearman correlation heatmap
df = smatrix
# 그림 사이즈 지정
fig, ax = plt.subplots( figsize=(15,15) )

# 삼각형 마스크를 만든다(위 쪽 삼각형에 True, 아래 삼각형에 False)
mask = np.zeros_like(df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 히트맵을 그린다
plt.rc('font', family= ourfont) 
sns.heatmap(df, 
            cmap = 'YlGnBu', 
            annot = True,   # 실제 값을 표시한다
            mask=mask,      # 표시하지 않을 마스크 부분을 지정한다
            linewidths=.5,  # 경계면 실선으로 구분하기
            cbar_kws={"shrink": .5},# 컬러바 크기 절반으로 줄이기
            vmin = -1,vmax = 1   # 컬러바 범위 -1 ~ 1
           )  
plt.title('Spearman triangle Correlation Heatmap', fontsize=20)
plt.savefig('./public/files/spearman_corr_tri_heatmap_'+filename+'_.png')


# In[ ]:


#pairplot 그리기 
plt.rc('font', family= ourfont) 
plt.figure(figsize=(20,20))
sns.pairplot(data_na_remove, kind="scatter", hue="label", palette="Set2")
plt.title('Pairplot', fontsize=20)
plt.savefig('./public/files/pairplot_'+filename+'_.png')

import zipfile
os.chdir("./public/files/")
file_ls = ['pearson_corr_heatmap_'+filename+'_.png','spearman_corr_heatmap_'+filename+'_.png','pearson_corr_tri_heatmap_'+filename+'_.png','spearman_corr_tri_heatmap_'+filename+'_.png', 'pairplot_'+filename+'_.png']
with zipfile.ZipFile('corrrelation_'+filename+'_.zip', 'w') as corr_zip:
    for i in file_ls:
        corr_zip.write(i)
    corr_zip.close()
