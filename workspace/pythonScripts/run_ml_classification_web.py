# -*- coding: utf-8 -*- 
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import tree, svm, linear_model, neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#feature_list = ['경색']

filepath = sys.argv[1]
filename = sys.argv[2]

data = pd.read_csv(filepath + "/" + filename, encoding='UTF-8')
#data = pd.read_csv("delete_50persent_NA_" + feature +".csv")

printResult = str({"filePath": filepath, "resultFileName": "result_"+filename+".csv"})
print(printResult.encode('utf-8'))
resultFile = filepath + "/" + "result_" + filename + ".csv"

data = data[data.label != 99]
label_list = data['label'].unique()
final = pd.DataFrame()
imp_mean = Imputer(strategy = 'median', axis = 0)

num = 0
for i in label_list :
	tmp = data[data.label == i]
	na_list = tmp.columns[tmp.isna().all()]
	for j in range(len(na_list)) :
		del data[na_list[j]]
		del tmp[na_list[j]]
		if num != 0 :
			del final[na_list[j]]
	imp_mean.fit(tmp)
	tmp_imp = imp_mean.transform(tmp)
	tmp_imp = pd.DataFrame(tmp_imp, columns = tmp.columns)
	final = pd.concat([final, tmp_imp], axis = 0)
	num = num + 1

#label = final['label']
#del final['label']

x_data_final = pd.DataFrame()
y_data_final = pd.DataFrame()
x_test_final = pd.DataFrame()
y_test_final = pd.DataFrame()

for i in label_list :
	tmp = final[final.label == i]
	label = tmp['label']
	del tmp['label']
	x_data, x_test, y_data, y_test = train_test_split(tmp, label, test_size=0.3, random_state=42)
	x_data_final = pd.concat([x_data_final, x_data], axis = 0)
	y_data_final = pd.concat([y_data_final, y_data], axis = 0)
	x_test_final = pd.concat([x_test_final, x_test], axis = 0)
	y_test_final = pd.concat([y_test_final, y_test], axis = 0)

y_data_final =  np.ravel(y_data_final, order = 'C')

#print(feature)

svm_classifier = svm.SVC(kernel = 'linear', random_state=0)
svm_classifier = svm_classifier.fit(x_data_final, y_data_final)
svm_predicted_class = svm_classifier.predict(x_test_final)
acc = accuracy_score(y_test_final, svm_predicted_class)

f = open(resultFile, 'w')
data = "SVM,%.4f\n" % acc
f.write(data)

rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier = rf_classifier.fit(x_data_final, y_data_final)
rf_predicted_class = rf_classifier.predict(x_test_final)
acc = accuracy_score(y_test_final, rf_predicted_class)

data = "RF,%.4f\n" % acc
f.write(data)

#dt_classifier = tree.DecisionTreeClassifier(random_state=0)
#dt_classifier = dt_classifier.fit(x_data_final, y_data_final)
#dt_predicted_class = dt_classifier.predict(x_test_final)
#acc = accuracy_score(y_test_final, dt_predicted_class)
#print("DT", acc)

naivebayes_classifier = GaussianNB()
naivebayes_classifier = naivebayes_classifier.fit(x_data_final, y_data_final)
naivebayes_predicted_class = naivebayes_classifier.predict(x_test_final)
acc = accuracy_score(y_test_final, naivebayes_predicted_class)

data = "NB,%.4f\n" % acc
f.write(data)

lr_classifier = LogisticRegression(random_state=0).fit(x_data_final, y_data_final)
lr_predicted_class = lr_classifier.predict(x_test_final)
acc = accuracy_score(y_test_final, lr_predicted_class)

data = "LR,%.4f\n" % acc
f.write(data)

knn_classifier = neighbors.KNeighborsClassifier(n_neighbors = len(label.unique()))
knn_classifier = knn_classifier.fit(x_data_final, y_data_final)
knn_predicted_class = knn_classifier.predict(x_test_final)
acc = accuracy_score(y_test_final, knn_predicted_class)

data = "KNN,%.4f\n" % acc
f.write(data)
f.close()

#print("###10cv###")
#label = np.ravel(final['label'], order = 'C')
#label_num = len(final['label'].unique())
#del final['label']
#svm_classifier = svm.SVC(kernel = 'linear', random_state=0)
#scores = cross_val_score(svm_classifier, final, label, cv=10)
#print("SVM", scores.mean())

#rf_classifier = RandomForestClassifier(random_state=0)
#scores = cross_val_score(rf_classifier, final, label, cv=10)
#print("RF", scores.mean())

#naivebayes_classifier = GaussianNB()
#scores = cross_val_score(naivebayes_classifier, final, label, cv=10)
#print("NB", scores.mean())

#lr_classifier = LogisticRegression(random_state=0)
#scores = cross_val_score(lr_classifier, final, label, cv=10)
#print("LR", scores.mean())

#knn_classifier = neighbors.KNeighborsClassifier(n_neighbors = label_num)
#scores = cross_val_score(knn_classifier, final, label, cv=10)
#print("KNN", scores.mean())
