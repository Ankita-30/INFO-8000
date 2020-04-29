# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 17:21:55 2020

@author: ar54482
"""

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression


white=pd.read_csv(r"winequality-white.csv", sep=';')
red=pd.read_csv(r"winequality-red.csv", sep=';')

#add the wine labels in the datasets
white['label']='white'
red['label']='red'

#combine the two datasets
wine=pd.concat([white,red])

#View your dataset
print(wine.info())


##visualize your data to determine if there is linearity between quality and the variables
#p=sns.pairplot(wine)
#pr=sns.pairplot(red)
#pw=sns.pairplot(white)

#No variables show a clear linear relationship hence, we will try to fit a model of multiple linear regression 
#between quality and the other variables

#Model1 for red wine
#Define the linear regression variables
Y1=red['quality']
X1=red[['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]

#LinearRegression1
lr1=LinearRegression()
lr1_fit=lr1.fit(X1,Y1)
y1pred=lr1_fit.predict(X1)
error1=Y1-y1pred
sse1=np.sum(error1*2)
rmse1=np.sqrt(sse1/len(Y1))


#Model2 for white wine
#Define the linear regression variables
Y2=white['quality']
X2=white[['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]

#LinearRegression2
lr2=LinearRegression()
lr2_fit=lr2.fit(X1,Y1)
y2pred=lr2_fit.predict(X2)
error2=Y2-y2pred
sse2=np.sum(error2*2)
rmse2=np.sqrt(sse2/len(Y2))


#Model3 for combined red & white 
#Define the linear regression variables
Y3=wine['quality']
X3=wine[['fixed acidity', 'volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]

#LinearRegression3
lr3=LinearRegression()
lr3_fit=lr3.fit(X1,Y1)
y3pred=lr3_fit.predict(X3)
error3=Y3-y3pred
sse3=np.sum(error3*2)
rmse3=np.sqrt(sse3/len(Y3))


#Visualize the residuals
f=plt.figure(figsize=(10,5))

#Model1
ax1=f.add_subplot(321)
ax1.hist(error1)

#Model2
ax2=f.add_subplot(322)
ax2.hist(error2)

#Model3
ax3=f.add_subplot(323)
ax3.hist(error3)


#Logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


#Logistic Regression

import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

Data=pd.read_csv(r"C:\Users\ar54482\Downloads\haberman.csv", sep=',')
Data.columns = ["Age","Year","Nodes","Survival"] #Add the header
print(Data.info())

#Define the input variables
X=Data.drop(columns=['Survival'])
y=Data['Survival']

#Visualize the data
sns.pairplot(Data)

#Build the model
LG=LogisticRegression(solver='lbfgs')
LG.fit(X,y)
y_pred=LG.predict(X)

print("The co-efficients and the intercept are as follows:")
print(str(LG.coef_) + "," + str(LG.intercept_))

print("The confusion matrix is as follows:")
print(confusion_matrix(y,y_pred))

