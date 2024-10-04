import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r'C:\Users\Admin\Desktop\House price prediction ml\kc_house_data.csv')
data.head()

data.shape

data.drop('id',axis=1,inplace=True)

data.info()

data.describe()

data.columns

data.drop(['zipcode', 'lat', 'long'],axis=1,inplace=True)

data.isnull().sum()

plt.figure(figsize=(5,5))
sns.histplot(data['price'])

import plotly.express as px
px.histogram(data['price'])


import plotly.express as px
px.box(data['price'])

plt.figure()
sns.countplot(x=data['bedrooms'])

plt.figure()
sns.scatterplot(x=data['bedrooms'],y=data['price'])

data['age']=data['yr_built'].apply(lambda x:2024-x)
plt.figure()
sns.scatterplot(x=data['age'],y=data['price'])

plt.figure()
sns.scatterplot(x=data['sqft_living'],y=data['price'])


data.drop('date',axis=1,inplace=True)

x=data.drop('price',axis=1)
y=data['price']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(x_train,y_train)

LR.coef_

LR.intercept_  #a

y_pred=LR.predict(x_test)

pd.DataFrame({'actual':y_test,'predicted':y_pred})

from sklearn import metrics
metrics.mean_squared_error(y_test,y_pred)

LR.score(x_test,y_test)
