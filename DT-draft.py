#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:32:46 2019

@author: wangjiatao
"""
# In[]
import pandas as pd
from sklearn import tree
import graphviz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# In[]

#data from websit: https://tianchi.aliyun.com/dataset/dataDetail?dataId=35769, its copyright belong data provider.
#There are 25 variables:
#1 ID: ID of each client
#2 LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
#3 SEX: Gender (1=male, 2=female)
#4 EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
#5 MARRIAGE: Marital status (1=married, 2=single, 3=others)
#6 AGE: Age in years
#7 PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
#8PAY_2: Repayment status in August, 2005 (scale same as above)
#9 PAY_3: Repayment status in July, 2005 (scale same as above)
#10 PAY_4: Repayment status in June, 2005 (scale same as above)
#11 PAY_5: Repayment status in May, 2005 (scale same as above)
#12 PAY_6: Repayment status in April, 2005 (scale same as above)
#13 BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
#14 BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
#15 BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
#16 BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
#17 BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
#18 BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
#19 PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
#20 PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
#21 PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
#22 PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
#23 PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
#24 default.payment.next.month: Default payment (1=yes, 0=no)
#imbalance data, so faulty

csv_file_path='files.csv'
cre_de=pd.read_csv(csv_file_path)
cre_de.head()

# In[]
datainput1=cre_de.iloc[:,0:22]
dataoutput1=cre_de.iloc[:,23]
x,x_test,y,y_test=train_test_split(datainput1,dataoutput1,test_size=0.2)

print(datainput1.columns)
print(datainput1.head())
# In[]
#fit & test
depthrange=range(40)
depth=1
scoreop=[]
for d in depthrange:
    clf=tree.DecisionTreeClassifier(criterion='gini',max_depth=depth,min_samples_split=10)
    clf=clf.fit(x,y)

    score1=clf.score(x_test,y_test)

    #print(score1);
    
    scoreop.append(score1)
    d=d+1
    depth=depth+1;
    
    
# In[]
plt.plot(scoreop)
scoreop=pd.DataFrame(scoreop)
max_op=scoreop.max()
print(scoreop,max_op)

# In[]

clf1=tree.DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=10)
clf1=clf1.fit(x,y)
scoreop7=clf1.score(x_test,y_test)
print("when depth =5, the score is",scoreop7)
#print(clf1.max_depth)
# In[]

dot_data=tree.export_graphviz(clf1,out_file=None)
graph=graphviz.Source(dot_data)
graph.render("credlimitdp5")

# In[]
#predict

#clf.predict([[  ]])
