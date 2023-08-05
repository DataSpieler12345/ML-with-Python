#!/usr/bin/env python
# coding: utf-8

# ## Breast Cancer Classification using Machine Learning | Machine Learning Projects

# * https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

# ### Importing the Dependencies 

# In[23]:


import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# ### Data Collection & Processing 

# ##### loading the data from sklearn

# In[2]:


breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)


# ##### loading the data to a pandas data frame 

# In[4]:


data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)


# In[5]:


data_frame.head()


# ##### adding the 'target' column to the panda data frame 

# In[6]:


data_frame['label'] = breast_cancer_dataset.target


# ##### print the last 5 rows of the dataframe

# In[8]:


data_frame.tail()


# In[9]:


#rows and columns in the dataset
data_frame.shape


# In[10]:


#getting some information about the data 
data_frame.info()


# ##### checking for missing values 
# 

# In[13]:


data_frame.isnull().sum()


# ##### statistical measures about the data

# In[12]:


data_frame.describe()


# ##### checking the distribution  of Target variable
# 
# * 1-->Bening
# 
# * 2-->Malignant

# In[14]:


data_frame['label'].value_counts()


# In[15]:


data_frame.groupby('label').mean()


# #### Separating the features and target 

# In[16]:


X = data_frame.drop(columns='label',axis=1)
Y = data_frame['label']


# In[17]:


print(X)


# In[18]:


print(Y)


# ### Splitting the data into training data & Testing data

# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[20]:


print(X.shape,X_train.shape, X_test.shape)


# ### Model Training 

# ##### Logistic Regression

# In[21]:


model = LogisticRegression()


# ##### training the Logistic Regression Model using Training Data 

# In[24]:


model.fit(X_train, Y_train)


# ### Model Evaluation

# ##### Accuracy Score 

# In[25]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


# In[27]:


print('Accuracy on training data =', training_data_accuracy)


# In[28]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)


# In[29]:


print('Accuracy on training data =', test_data_accuracy)


# ### Building a Predictive System

# In[31]:


input_data = (18.25,19.98,119.6,1040,0.09463,0.109,0.1127,0.074,0.1794,0.05742,0.4467,0.7732,3.18,53.91,0.004314,0.01382,0.02254,0.01039,0.01369,0.002179,22.88,27.66,153.2,1606,0.1442,0.2576,0.3784,0.1932,0.3063,0.08368)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Breast cancer is Malignant')
    
else:
    print('The Breast cancer is Benign')


# In[32]:


input_data = (12.46,24.04,83.97,475.9,0.1186,0.2396,0.2273,0.08543,0.203,0.08243,0.2976,1.599,2.039,23.94,0.007149,0.07217,0.07743,0.01432,0.01789,0.01008,15.09,40.68,97.65,711.4,0.1853,1.058,1.105,0.221,0.4366,0.2075)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Breast cancer is Malignant')
    
else:
    print('The Breast cancer is Benign')


# In[33]:


input_data = (16.74,21.59,110.1,869.5,0.0961,0.1336,0.1348,0.06018,0.1896,0.05656,0.4615,0.9197,3.008,45.19,0.005776,0.02499,0.03695,0.01195,0.02789,0.002665,20.01,29.02,133.5,1229,0.1563,0.3835,0.5409,0.1813,0.4863,0.08633)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Breast cancer is Malignant')
    
else:
    print('The Breast cancer is Benign')


# In[34]:


input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Breast cancer is Malignant')
    
else:
    print('The Breast cancer is Benign')


# In[ ]:




