#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:54:46 2020

@author: suhas
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import statsmodels.api as sm

#importing dataset
df=pd.read_csv("/Users/suhas/Desktop/Fresh/MLR/insurance.csv")

#to check if thr are any null values
df.isnull().any()
df.isnull().sum()

#to find duplicated values
df.duplicated().sum()
df.drop_duplicates(inplace=True)


#to check column names
print(df.columns)
df.dtypes

#check first 5 value
df.head(5)
#df['region'].unique()

#We have 2 columns(sex,smoker) to be label encode and region column to one hot encode

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
'''
#way 1

gender={'female': 0,'male': 1}
df['sex']=[gender[item] for item in df['sex']]

#way 2
le.fit_transform(df['smoker'])

#way 3
#df['smoker'] = df['smoker'].replace({"yes":1,"no":0})
''''
# way 4
df[['sex','smoker']]=df[['sex','smoker']].apply(le.fit_transform)

#one hot encoding using pandas getdummies
dummy=pd.get_dummies(df['region'])
dummy=dummy.drop(['northeast'],axis=1)
df= pd.concat([df,dummy],axis=1)

#data for building model
df=df.drop(['region'],axis=1)

#let's visualize data
#distribution
sns.distplot(df['bmi'])
sns.distplot(df['expenses'])

# corelation
cor=df.corr()
sns.heatmap(cor,annot=True)

#BMi : normally distributed, expenses is right skewed,

#dividing the data into dependent and independent variable
y=df['expenses']
x=df.drop(['expenses'],axis=1)

#splitting of data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
# we can scale town the data but since regression model can handle this no need to scale

from sklearn.linear_model import LinearRegression
le=LinearRegression()
le.fit(x_train,y_train)

#predict
y_pred=le.predict(x_test)

# finding residulas for various check
residual=y_test-y_pred

# to check accuracy of model
score=le.score(x_test,y_test)
#accuracy 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
acc=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)

# model give 73.90% accuracy

# checking if assumptions of MLR is correct or not

#To check normality of residulas

#1.distplot of residulas
sns.distplot(residual)
#probdistribution
import scipy
fig, ax = plt.subplots(figsize=(6,2.5))
scipy.stats.probplot(residual,plot=ax)
#mean of residulas should be zero
np.mean(residual)
#qq plot
sm.qqplot(residual)

#Multicolinerity done with corr
#2 : VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif["VIF"]=[variance_inflation_factor(x.values,i)for i in range(x.shape[1])]
vif["features"]=x.columns


#age and BMi value are greater than 5, so multi colinearity is present

#To find if data is not heteroscadecity
plt.scatter(y_pred,residual)

# thr is no pattern and most of the data are in zero range

# To find if no autocorelation existes
from statsmodels.tsa.api import graphics as gp
gp.plot_acf(residual,lags=40,alpha=0.05)
acf.show()















