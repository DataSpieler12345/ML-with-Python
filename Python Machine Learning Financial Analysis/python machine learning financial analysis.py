#!/usr/bin/env python
# coding: utf-8

# <center><h1><b><font size="6">Python Machine Learning Financial Analysis</font></b></h1></center>

# #### Import Libraries

# In[9]:


import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt


# #### Load the data

# In[57]:


data = pd.read_excel('./data/default_of_credit_card_clients.xls', header=0)
data.head()


# #### Information about the different columns:
# 
# 
# 
# **ID**: Account ID
# 
# **LIMIT_BAL**: Credit limit (in New Taiwanese dollars (NT)) including both individual and family (supplementary) credit
# 
# **SEX**: 1 = male; 2 = female
# 
# **EDUCATION**: 1 = graduate school; 2 = university; 3 = high school; 4 = others
# 
# **MARRIAGE**: Marital status (1 = married; 2 = single; 3 = others)
# 
# **AGE**: Age
# 
# **PAY_1 - PAY_6**: History of past payments. History from April to September. The rating scale is as follows: -2 = No consumption; -1 = Paid in full; 0 = The use of revolving credit; 1 = Payment delay for one month; 2 = Payment delay for two months; and so on up to 9 = Payment delay for nine months and above
# 
# **BILL_AMT1 - BILL_AMT6**: Amount of bill statement. BILL_AMT1 represents the amount of the bill statement in September, and 
# BILL_AMT6 represents the amount of the bill statement in April.
# 
# **PAY_AMT1 - PAY_AMT6**: Amount of previous payment (in NT dollars).

# In[15]:


data.shape


# In[58]:


data.info()


# ##### unique ID Values in the dataset

# In[59]:


data['ID'].nunique()


# In[24]:


data['ID'].value_counts()


# In[25]:


id_counts =data['ID'].value_counts()
id_counts[:3] # the first 3 


# In[26]:


id_counts.value_counts()


# In[20]:


# boolean mask
bool_mask = id_counts == 2
bool_mask[:5]


# #### Exploring financial history

# In[60]:


data = pd.read_excel('./data/default_of_credit_card_clients.xls', header=0)
data.head()


# In[61]:


data.shape


# In[62]:


data.columns


# #### PAY Columns

# In[68]:


pay_columns = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
data[pay_columns].describe()


# #### allows you to see payment behavior, where most of the data and payment trends of credit card customers are concentrated. In this we know the maximum and minimum values, and it is observed that most of the customers are up to date with their payments since it reflects the highest concentration of data.

# In[69]:


data[pay_columns[0]].value_counts().sort_index()


# ##### Grafic

# In[71]:


data[pay_columns[0]].hist()


# ##### 0 = represents loan payments 
# ##### 1 = represents non-payments, overdue
# 

# In[72]:


data['default payment next month'].value_counts()     


# #### Every PAY column grafic
# 
# *we can see that from PAY_2 the information is not correct*.
# 

# In[73]:


import matplotlib as mlp
mlp.rcParams['figure.dpi'] = 400 # high definition
mlp.rcParams['font.size'] = 4 # text size
data[pay_columns].hist(layout=(2, 3))


# In[74]:


data.loc[data['PAY_2'] == 2, ['PAY_2', 'PAY_3']].head()


# ##### the others columns

# In[75]:


bill_feats = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
pay_amt_feats = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


# In[76]:


data[bill_feats].describe()


# #### customer receipts

# In[39]:


data[bill_feats].hist(layout=(2, 3))


# #### customer payments

# In[40]:


data[pay_amt_feats].hist(layout=(2, 3))


# In[41]:


pay_zero_mask = data[pay_amt_feats] == 0
pay_zero_mask.sum()


# ##### graphical payments representation without non-paying customers

# In[42]:


data[pay_amt_feats][~pay_zero_mask].apply(np.log10).hist(layout=(2, 3))


# #### credit limit column

# In[43]:


data['LIMIT_BAL'].hist()


# #### Machine Learning

# In[79]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[['EDUCATION', 'PAY_1', 'LIMIT_BAL']].values, #columns to use / independent
                                                    data['default payment next month'], #variable to predict / dependent
                                                    test_size=.2, random_state=24) # size 20% - of the data to testing - X_test + y_test / 80% X_train
# size 20% - of the data to testing - X_test + y_test / 80% X_train
# 80 % (X_train) of EDUCATION, PAY_1 & LIMIT_BAL
# 20 % (X_test) of EDUCATION, PAY_1 & LIMIT_BAL
# 80 % (y_train) of default payment next month column (to predict)
# 20 % (y_test) of default payment next month column (to predict)


# In[80]:


# 24000 rows to train the model
# 6000 rows to test the model

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ##### 0 = 23364 / paying customers
# 
# ##### 1 = 6636 / non-paying customers
# 
# there is no symmetry in the data, we must find an algorithm that adapts to this business need.

# In[47]:


data['default payment next month'].value_counts()


# #### RandomForestClassifier
# 
# This algorithm can handle scenarios well when there are large differences in the data or no symmetry, e.g. payments and non-payments.

# In[48]:


from sklearn.ensemble import RandomForestClassifier

bosque_aleatorio = RandomForestClassifier(n_estimators=100, random_state=24) #n_estimators = 100 = numbers of forest (the higher the number, the more accurate it can be, but the harder it is to train)

bosque_aleatorio.fit(X_train, y_train) #fit the model = values to get and train...

y_pred = bosque_aleatorio.predict(X_test) #variable to predict / prediction of values


# #### show the predicted values and compare it with the values of the y_test

# In[49]:


y_pred[:15]


# #### show the real values and compare it with the values of the y_pred

# In[50]:


y_test[:15]


# ##### Accuracy / ML / RandomForestClassifierModel
# 
# 81.15 % of the data. So we can say that the model is quite good since it used more than 80% of the data. However, we can add other columns of interest to the model and it can affect the accuracy of the model as well.

# In[51]:


from sklearn.metrics import accuracy_score

accury = accuracy_score(y_test, y_pred)
accury


# ##### LogisticRegression

# In[52]:


from sklearn import linear_model

logit_model = linear_model.LogisticRegression()
logit_model.fit(X_train, y_train)


# In[53]:


y_pred = logit_model.predict(X_test)


# In[54]:


y_pred[:15]


# In[55]:


y_test[:15]


# ##### Accuracy / ML / LogiticRegression

# In[56]:


accury = accuracy_score(y_test, y_pred)
accury


# ##### Conclusion

# The Logistic **Regression model (77.38% - accuracy)** is not very good for this data scenario unlike the **RandomForestClassifier model (81.15% - accuracy)**. 
