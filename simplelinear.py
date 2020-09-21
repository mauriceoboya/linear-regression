#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:01:45 2020

@author: numberphile
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('/home/numberphile/machine learning/salary prediction/salarydata.csv')

#getting the first five rows
print(df.head())

#separating the dataset into dependent and predictor
x= df.iloc[:, :-1].values  
y= df.iloc[:, 1].values 

#split the data into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(x, y, test_size=1/3, random_state=0)

#Fitting the Simple Linear Regression model to the training dataset 
from sklearn.linear_model import LinearRegression
reg= LinearRegression()  
reg.fit(x_train, y_train)  

#goodness of the test using r-squared
r2_score=reg.score(x_train, y_train)
print(r2_score)

#Prediction of Test and Training set result  
y_pred= reg.predict(x_test)  
x_pred= reg.predict(x_train)  

#predicting a salary expected with 17 years of experience
y_pred = reg.predict([[17]])
print(y_pred)

plt.scatter(x_train, y_train, color="green")   
plt.plot(x_train, x_pred, color="red")    
plt.title("Salary vs Experience (Training Dataset)")  
plt.xlabel("years")  
plt.ylabel("Salary")  
plt.show()   

#visualizing the Test set results  
plt.scatter(x_test, y_test, color="blue")   
plt.plot(x_train, x_pred, color="red")    
plt.title("Salary vs Experience (Test Dataset)")  
plt.xlabel("Years")  
plt.ylabel("Salary")  
plt.show()  