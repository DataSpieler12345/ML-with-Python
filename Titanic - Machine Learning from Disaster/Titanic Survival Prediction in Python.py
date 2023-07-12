#!/usr/bin/env python
# coding: utf-8

# ### Titanic Survival Prediction in Python - ML Project 
# 
# * https://www.kaggle.com/competitions/titanic/overview

# In[30]:


#Import the libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


#set style for sns pplots
sns.set()


# #### Loading the data set

# In[33]:


titanic_data = pd.read_csv('Data/train.csv')
titanic_data.head()


# #### StratifiedShuffleSplit

# In[34]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_indices, test_indices in split.split(titanic_data, titanic_data[["Survived", "Pclass", "Sex"]]):
    strat_train_set = titanic_data.loc[train_indices]
    strat_test_set = titanic_data.loc[test_indices]


# In[35]:


#strat_test_set
strat_test_set


# In[38]:


#a plot 
plt.subplot(1,2,1)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.subplot(1,2,2)
strat_test_set['Survived'].hist()
strat_test_set['Pclass'].hist()

plt.show()
#notice that we have equal distribution
#blue bar = Survived
#Orange bar = Not Survived


# #### Missing values 

# In[39]:


strat_train_set.info()
# we have missing values 
# the age has missing values we need to impute it 
# Name same situation / impute...


# #### BaseEstimator | TransformerMixin

# In[50]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class AgeImputer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Age'] = imputer.fit_transform(X[['Age']])
        return X


# #### One hot encoding 

# In[53]:


from sklearn.preprocessing import OneHotEncoder

class FeatureEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Embarked']]).toarray()
        
        column_names = ["C", "S", "Q", "N"]
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        
        matrix = encoder.fit_transform(X[['Sex']]).toarray() 
        
        column_names = ["Female", "Male"]
                      
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
            
        return X    


# #### Dropp some features that are not relevant 
# 

# In[54]:


class FeatureDropper(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(["Embarked","Name","Ticket","Cabin","Sex","N"], axis=1, errors="ignore")


# ### Define the pipeline 

# In[56]:


from sklearn.pipeline import Pipeline 

#the object pipeline
pipeline = Pipeline ([("ageimputer", AgeImputer()),
                      ("featureencoder", FeatureEncoder()),
                      ("featuredropper", FeatureDropper())])


# #### Call the fit transform function 

# In[57]:


#all info of all the estimator called
strat_train_set = pipeline.fit_transform(strat_train_set)


# In[59]:


#see the strat_train_set data
strat_train_set.head()


# In[60]:


#see the strat_train_set.info()
strat_train_set.info()
#we dont have non-null values and any NaN values anymore 


# ### Scaling the data

# In[62]:


from sklearn.preprocessing import StandardScaler

X = strat_train_set.drop(['Survived'], axis= 1)
y = strat_train_set['Survived']

scaler = StandardScaler()
X_data = scaler.fit_transform(X)
y_data = y.to_numpy()


# ### Random Forest Classifier Algorithm

# In[66]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#crossvalidation
#define the classifier to be a random forest classifier
clf = RandomForestClassifier()

#define the paramter grid, a dictionary or a list of dictionaries
param_grid = [
    {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2,3,4]}
]

grid_search = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data, y_data)


# #### get the best estimator
# 

# In[67]:


final_clf = grid_search.best_estimator_


# In[68]:


final_clf
#best depth = 5, n_estimators=200


# In[69]:


#take the test data set that we have | run the hole process of the pre-processing on it again
strat_test_set = pipeline.fit_transform(strat_test_set)


# In[71]:


strat_test_set.head()


# In[72]:


#score the test data set
X_test = strat_test_set.drop(['Survived'], axis=1)
y_test = strat_test_set['Survived']

scaler = StandardScaler()
X_data_test = scaler.fit_transform(X_test)
y_data_test = y_test.to_numpy()


# #### Evalutation

# In[73]:


final_clf.score(X_data_test, y_data_test)
#84% accuracy 


# #### take again all the data and train de model again

# In[74]:


# combine mixed data = titanic_data through the pipeline, because the titanic data still has the old format
titanic_data


# In[75]:


#we can say that will be our final data
final_data = pipeline.fit_transform(titanic_data)


# In[77]:


final_data
#notice: final data with the right format


# In[78]:


#score the  final data set
X_final = final_data.drop(['Survived'], axis=1)
y_final = final_data['Survived']

scaler = StandardScaler()
X_data_final = scaler.fit_transform(X_final)
y_data_final = y_final.to_numpy()


# In[80]:


#crossvalidation
#define the classifier to be a random forest classifier
prod_clf = RandomForestClassifier()

#define the paramter grid, a dictionary or a list of dictionaries
param_grid = [
    {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2,3,4]}
]

grid_search = GridSearchCV(prod_clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data_final, y_data_final)


# #### get the best estimator

# In[81]:


prod_final_clf = grid_search.best_estimator_


# In[82]:


prod_final_clf


# #### Making Predictions

# In[83]:


titanic_test_data = pd.read_csv('Data/test.csv')


# In[84]:


titanic_test_data
#it has the same structure but not the Survived column


# In[85]:


#pipeline with test.csv
final_test_data = pipeline.fit_transform(titanic_test_data)


# In[86]:


final_test_data


# In[88]:


final_test_data.info()
#missing values =Fare column


# In[92]:


X_final_test = final_test_data
X_final_test = X_final_test.fillna(method="ffill")

scaler = StandardScaler()
X_data_final_test = scaler.fit_transform(X_final_test)


# #### Predictions

# In[93]:


predictions = prod_final_clf.predict(X_data_final_test)


# In[94]:


#look at the predictions created
predictions


# ### create the final data frame with the predictions created 

# In[95]:


final_df = pd.DataFrame(titanic_test_data['PassengerId'])
final_df['Survived'] = predictions
#store it into a csv file named predictions
final_df.to_csv("Data/predictions.csv", index=False)


# In[96]:


#show the result
final_df


# ### Made-by.dataspieler12345
# 
