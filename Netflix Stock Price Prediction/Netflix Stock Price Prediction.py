#!/usr/bin/env python
# coding: utf-8

# # Netflix Stock Price Prediction

# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,r2_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')


# ### Data Loading

# In[3]:


df = pd.read_csv('data/NFLX.csv')


# In[4]:


df.head()


# In[5]:


#DF data Viz
viz = df.copy()
viz.head()


# ### Data Preparation

# In[6]:


df.isnull().sum()


# In[8]:


df.shape


# In[9]:


df.info()


# In[11]:


df.describe().T


# ### Split the data into Train&Test set

# In[13]:


train, test = train_test_split(df, test_size = 0.2)


# In[14]:


test_pred = test.copy()


# In[15]:


train.head(10)


# In[16]:


test.head(10)


# In[17]:


x_train = train[['Open', 'High', 'Low', 'Volume']].values
x_test = test[['Open', 'High', 'Low', 'Volume']].values


# In[18]:


y_train = train['Close'].values
y_test = test['Close'].values


# ### Linear Regression

# In[19]:


model_lnr = LinearRegression()
model_lnr.fit(x_train, y_train)


# In[20]:


y_pred = model_lnr.predict(x_test)


# In[21]:


result = model_lnr.predict([[262.000000, 267.899994, 250.029999, 11896100]])
print(result)


# ### Model Evaluation

# In[23]:


print("MSE",round(mean_squared_error(y_test,y_pred), 3))
print("RMSE",round(np.sqrt(mean_squared_error(y_test,y_pred)), 3))
print("MAE",round(mean_absolute_error(y_test,y_pred), 3))
print("MAPE",round(mean_absolute_percentage_error(y_test,y_pred), 3))
print("R2 Score : ", round(r2_score(y_test,y_pred), 3))


# ### Model Visualization

# In[25]:


def style():
    plt.figure(facecolor='black', figsize=(15,10))
    ax = plt.axes()

    ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to white
    ax.tick_params(axis='y', colors='white')    #setting up Y-axis tick color to white

    ax.spines['left'].set_color('white')        #setting up Y-axis spine color to white
    #ax.spines['right'].set_color('white')
    #ax.spines['top'].set_color('white')
    ax.spines['bottom'].set_color('white')      #setting up X-axis spine color to white

    ax.set_facecolor("black")                   # Setting the background color of the plot using set_facecolor() method


# In[26]:


viz['Date']=pd.to_datetime(viz['Date'],format='%Y-%m-%d')


# In[27]:


data = pd.DataFrame(viz[['Date','Close']])
data=data.reset_index()
data=data.drop('index',axis=1)
data.set_index('Date', inplace=True)
data = data.asfreq('D')
data


# In[28]:


style()

plt.title('Closing Stock Price', color="white")
plt.plot(viz.Date, viz.Close, color="#94F008")
plt.legend(["Close"], loc ="lower right", facecolor='black', labelcolor='white')


# In[29]:


style()

plt.scatter(y_pred, y_test, color='red', marker='o')
plt.scatter(y_test, y_test, color='blue')
plt.plot(y_test, y_test, color='lime')


# In[30]:


test_pred['Close_Prediction'] = y_pred
test_pred


# In[31]:


test_pred[['Close', 'Close_Prediction']].describe().T


# ### Savingo the Data as CSV
# 

# In[35]:


test_pred['Date'] = pd.to_datetime(test_pred['Date'],format='%Y-%m-%d')


# In[36]:


output = pd.DataFrame(test_pred[['Date', 'Close', 'Close_Prediction']])
output = output.reset_index()
output = output.drop('index',axis=1)
output.set_index('Date', inplace=True)
output =  output.asfreq('D')
output


# In[37]:


output.to_csv('Close_Prediction.csv', index=True)
print("CSV successfully saved!")

