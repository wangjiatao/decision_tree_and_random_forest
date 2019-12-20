#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:01:48 2019

@author: wangjiatao
"""

# In[]
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt



# In[]

csv_file_path='files.csv'
cre_de=pd.read_csv(csv_file_path)
datainput1=cre_de.iloc[:,0:22]
dataoutput1=cre_de.iloc[:,23]
x,x_test,y,y_test=train_test_split(datainput1,dataoutput1,test_size=0.2)



# In[]
#find optimal estimator
estnum=2
scores=[]
dsa=1
for das in range(25):
    rf1=RandomForestClassifier(n_estimators=estnum,criterion='gini', max_depth=5)
    rf1=rf1.fit(datainput1,dataoutput1)
    estnum=estnum+1
    dsa=dsa+1
    scores_rf = cross_val_score(rf1, x_test, y_test, cv=5)
    scoresm=scores_rf.mean()
    scores.append(scoresm);
    
    
    

# In[]
#drawning
scores=pd.DataFrame(scores)
    
print(scores)
print(scores.max())
print(scores.idxmax())

plt.plot(scores)

opest=int(scores.idxmax())

print("the optimal estimator is" ,opest)

# In[]
#apple optimal estimator
rfop=RandomForestClassifier(n_estimators=opest,criterion='gini', max_depth=5)
rfop=rfop.fit(datainput1,dataoutput1)
scores = cross_val_score(rfop, x_test, y_test, cv=5)
print(scores.mean())


# In[]
#predict
preder=[[78000,1,2,1,31,-1,-1,-1,-1,-1,-1,5789,4421,3089,
        5967,2102,3799,3089,5967,2102,3799,4101]]


default=rfop.predict(preder)
print(default)

if default==[0]:
    print("随机森林预测其下个月不会违约")
else:
    print("下个月会违约")

