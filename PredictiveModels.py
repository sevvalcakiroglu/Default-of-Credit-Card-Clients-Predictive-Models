# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:05:47 2019

@author: sevval.cakiroglu
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import cross_val_score



df = pd.read_excel('defaultofcreitcardclients.xls')
df.describe()

total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()


plt.figure(figsize = (14,6))
plt.title('Amount of credit limit - Density Plot')
sns.set_color_codes("pastel")
sns.distplot(df['LIMIT_BAL'],kde=True,bins=200, color="blue")
plt.show()


df['LIMIT_BAL'].value_counts().head(5)


corr=df.corr()
sns.heatmap(corr)

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
df = df[selected_columns]

list=['ID','default payment next month']
y=df['default payment next month']
X=df.drop(list, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.head()

scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#KNN 
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
print('Knn Train Score:', knn.score(X_train,y_train))
print('Knn Test Score:', knn.score(X_test,y_test))

#GridSearchCV
grid_params={'n_neighbors':[3,5,11,19],
            'weights':['uniform','distance'],
            'metric':['euclidean','manhattan']}

gs = GridSearchCV(knn,
                 grid_params,
                 cv=3,
                 )
gs_results = gs.fit(X_train, y_train)
print('En iyi score',gs_results.best_score_)
print('En iyi parametreler',gs_results.best_params_)
print('Knn Train Yeni Score:', gs.score(X_train,y_train))
print('Knn Test Yeni Score:', gs.score(X_test,y_test))

#Prediction
y_pred=gs.predict(X_test)
print(classification_report(y_test,y_pred))
cm1=confusion_matrix(y_test,y_pred)
print('KNeighborsClassifier confusion matrix:\n', cm1)

#RandomForest
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
print('RandomForestClassifier Train Score:', rfc.score(X_train,y_train))
print('RandomForestClassifier Test Score:', rfc.score(X_test, y_test))

 #GridsearchCV
parameters = {
    'n_estimators'      : [50,100,200],
    'max_depth'         : [8, 9, 10, 11, 12],
    'random_state'      : [0],
}

gs1 = GridSearchCV(rfc,
                 parameters,
                 cv=3,
                 )
gs1_results = gs1.fit(X_train, y_train)
print('En iyi score',gs1_results.best_score_)
print('En iyi parametreler',gs1_results.best_params_)
print('RandomForestClassifier Train Yeni Score:', gs1.score(X_train,y_train))
print('RandomForestClassifier Test Yeni Score:', gs1.score(X_test,y_test))

#prediction
y_pred1=gs1.predict(X_test)
print(classification_report(y_test,y_pred1))
cm2=confusion_matrix(y_test,y_pred1)
print('RandomForestClassifier confusion matrix:\n', cm2)


#SVC prediction
svc=SVC(kernel='linear', C=1)
svc.fit(X_train,y_train)
svc_pred=svc.predict(X_test)
print('SVC Train Score:', svc.score(X_train,y_train))
print('SVC Test Score:', svc.score(X_test, y_test))

print(classification_report(y_test,svc_pred))
cm3=confusion_matrix(y_test,svc_pred)
print('SVC confusion matrix:\n', cm3)


#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train,y_train)
print('GaussianNB Train Score:', gnb.score(X_train,y_train))
print('GaussianNB Test Score:', gnb.score(X_test, y_test))
y_pred2=gnb.predict(X_test)

print(classification_report(y_test,y_pred2))
cm4=confusion_matrix(y_test,y_pred2)
print('GaussianNB confusion matrix:\n', cm4)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
print('LogisticRegression Train Score:', log_reg.score(X_train, y_train))
print('LogisticRegression Test Score:', log_reg.score(X_test, y_test))
y_pred3=log_reg.predict(X_test)
print(classification_report(y_test,y_pred3))
cm5=confusion_matrix(y_test,y_pred3)
print('LogisticRegression confusion matrix:\n', cm5)

 #XGBoost
xgb=XGBClassifier()
xgb.fit(X_train,y_train)
print('Knn Train Score:', xgb.score(X_train,y_train))
print('Knn Test Score:', xgb.score(X_test,y_test))

#GridSearchCV
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
grids = GridSearchCV(xgb,
                 params,
                 cv=3,
                 )
grids_results = grids.fit(X_train, y_train)
print('En iyi score',grids_results.best_score_)
print('En iyi parametreler',grids_results.best_params_)
print('XGBClassifier Train Yeni Score:', grids.score(X_train,y_train))
print('XGBClassifier Test Yeni Score:', grids.score(X_test,y_test))

#prediction
y_predd=grids.predict(X_test)
print(classification_report(y_test,y_predd))
cm6=confusion_matrix(y_test,y_predd)
print('XGBClassifier confusion matrix:\n', cm6)









