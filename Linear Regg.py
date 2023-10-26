# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:54:58 2023

@author: HP
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("C:/Users/HP/Downloads/Ecommerce Customers (1).csv")

df.head()
df.dtypes

q=df.describe()

#exploratory DA
sns.jointplot(x="Time on App",y="Yearly Amount Spent",data=df)

sns.pairplot(df)

x=df.iloc[:,3:7].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)

#prediction
pred_=lm.predict(X_test)

import sklearn.metrics as metrics
metrics.mean_absolute_error(y_test, pred_)
metrics.mean_squared_error(y_test,pred_)
np.sqrt(metrics.mean_squared_error(y_test,pred_))



plt.scatter(y_test,pred_)
plt.xlabel("Predicted")
plt.ylabel("y test")

#finding the coefficient of features/ large value better dependency
lm.coef_
pd.DataFrame(lm.coef_,index=["Avg. Session Length",
"Time on App",             
"Time on Website",         
"Length of Membership"])
