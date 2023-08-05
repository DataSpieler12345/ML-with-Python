#!/usr/bin/env python
# coding: utf-8

# # German Credit Risk Classification

# ## Content

# ##### The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes.

# ### The following steps were followed in this project:

# 1. Import Modules and Data
# 2. Data Analysis
# 3. Data Classification
# 4. Data Visualization
# 5. Data Preprocessing
# 6. Building Models
# 	* DecisionTree Model
# 	* GradientBoosting Model
# 	* XGBoost Model
# 	* LightGBM Model

# * link_data = https://www.kaggle.com/kabure/german-credit-data-with-risk?select=german_credit_data.csv

# ## Import Modules and Data
# 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


# In[41]:


df = pd.read_csv("data/german_credit_risk.csv")
df = df.iloc[:,1:]
df.head()


# In[42]:


df.dtypes


# ## Variable Description

# Meaning of the Values:
# 
# 1. Age: Age of the person applying for the credit.
# 2. Sex: Gender of the person applying for the credit.
# 3. Job: 0,1,2,3 The values specified for the job in the form of 0,1,2,3.
# 4. Housing: own, rent or free.
# 5. Saving accounts: the amount of money in the person's bank account.
# 6. Checking account: cheque account.
# 7. Credit amount: Credit amount.
# 8. Duration: Time given for credit payment.
# 9. Purpose: Goal of credit application.
# 10. Risk:Credit application positive or negative.

# ## Data Analysis

# In[43]:


df.info()


# In[44]:


df.describe()


# In[45]:


#Only numeric columns
numeric_columns = df.select_dtypes(include=[np.number])
correlation = numeric_columns.corr()


# In[46]:


print(correlation)


# In[47]:


df.dtypes


# ### Unique Values

# In[48]:


def unique_value(data_set, column_name):
    return data_set[column_name].nunique()

print("Number of the Unique Values:")
print(unique_value(df,list(df.columns)))


# ### Checking missing values 

# In[49]:


def missing_value_table(df):
    missing_value = df.isna().sum().sort_values(ascending=False)
    missing_value_percent = 100 * df.isna().sum()//len(df)
    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)
    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})
    cm = sns.light_palette("lightgreen", as_cmap=True)
    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)
    return missing_value_table_return
  
missing_value_table(df)


# ### Sex Distribution

# In[50]:


pd.crosstab(df["Sex"],df["Risk"])


# ### Housing Distribution

# In[51]:


pd.crosstab(df["Housing"],df["Risk"])


# * There is multiple groups in the "Purpose".
# * At this situation we can apply ANOVA test.
# * This way we will see the differences according to requisition of Credit Amount.

# In[52]:


df.head(2)


# In[53]:


from scipy import stats

df1 = df.copy()

df1 = df1[["Credit amount","Purpose"]]

group = pd.unique(df1.Purpose.values)

d_v1 = {grp:df1["Credit amount"][df1.Purpose == grp] for grp in group}


# * One of conditions ANOVA test is equal variance.
# * Applied levene and according to result, between groups variances are not equal.

# ### Applying levene

# In[55]:


stats.levene(d_v1['radio/TV'],d_v1['furniture/equipment'],d_v1['car'],d_v1['business'],d_v1['domestic appliances'],d_v1['repairs'],
                     d_v1['vacation/others'],d_v1['education'])


# * P value << 0.05

# In[56]:


f, p = stats.f_oneway(d_v1['radio/TV'],d_v1['furniture/equipment'],d_v1['car'],d_v1['business'],d_v1['domestic appliances'],d_v1['repairs'],
                     d_v1['vacation/others'],d_v1['education'])

("F statistics: "+str(f)+" | P value : "+str(p))


# * H0: There are no significant differences means of groups.
# 
# * H1: At least one group's mean is different.
# 
# * P value < 0.05
# 
# * Reject h0 hypothesis

# In[57]:


(df.groupby(by=["Purpose"])[["Credit amount"]].agg("sum") / df["Credit amount"].sum())*100


# * In the result, there is different between groups.
# * In this query we can see difference

# ## Data Classification

# In[87]:


sns.set(font_scale=1,style="whitegrid")
fig,ax=plt.subplots(ncols=2,nrows=3,figsize=(16,12))
cat_list=["Age","Credit amount","Duration"]
count=0
for i in range(3):
    sns.distplot(df[cat_list[count]],ax=ax[i][0],kde=False,color="#F43EEC")
    sns.kdeplot(df[cat_list[count]],ax=ax[i][1],shade=True,color="#359F4B")
    count+=1


# * "monthly pay" and "credit amount^2"(square) added in data frame.

# In[88]:


df["Monthly pay"] = (df["Credit amount"] / df["Duration"])
df["Credit amount^2"] = df["Credit amount"]**2


# * 'Age' and 'Duration' columns Classification

# In[89]:


df.insert(1,"Cat Age",np.NaN)
df.loc[df["Age"]<25,"Cat Age"]="0-25"
df.loc[((df["Age"]>=25) & (df["Age"]<30)),"Cat Age"]="25-30"
df.loc[((df["Age"]>=30) & (df["Age"]<35)),"Cat Age"]="30-35"
df.loc[((df["Age"]>=35) & (df["Age"]<40)),"Cat Age"]="35-40"
df.loc[((df["Age"]>=40) & (df["Age"]<50)),"Cat Age"]="40-50"
df.loc[((df["Age"]>=50) & (df["Age"]<76)),"Cat Age"]="50-75"


# In[ ]:


df.insert(9,"Cat Duration",df["Duration"])
for i in df["Cat Duration"]:
    if i<12:
        df["Cat Duration"]=df["Cat Duration"].replace(i,"0-12")
    elif (i>=12) and (i<24):
        df["Cat Duration"]=df["Cat Duration"].replace(i,"12-24")
    elif (i>=24) and (i<36):
        df["Cat Duration"]=df["Cat Duration"].replace(i,"24-36")
    elif (i>=36) and (i<48):
        df["Cat Duration"]=df["Cat Duration"].replace(i,"36-48")
    elif (i>=48) and (i<60):
        df["Cat Duration"]=df["Cat Duration"].replace(i,"48-60")
    elif (i>=60) and (i<=72):
        df["Cat Duration"]=df["Cat Duration"].replace(i,"60-72")


# In[62]:


df.insert(4,"Cat Job",df["Job"])
df["Cat Job"]=df["Cat Job"].astype("category")
df["Cat Job"]=df["Cat Job"].replace(0,"unskilled")
df["Cat Job"]=df["Cat Job"].replace(1,"resident")
df["Cat Job"]=df["Cat Job"].replace(2,"skilled")
df["Cat Job"]=df["Cat Job"].replace(3,"highly skilled")


# In[63]:


df["Job"]=pd.Categorical(df["Job"],categories=[0,1,2,3],ordered=True)
df["Cat Age"]=pd.Categorical(df["Cat Age"],categories=['0-25','25-30', '30-35','35-40','40-50','50-75'])
df["Cat Duration"]=pd.Categorical(df["Cat Duration"],categories=['0-12','12-24', '24-36','36-48','48-60','60-72'])


# In[91]:


df.head()


# ## Data Visualization

# In[92]:


fig, ax = plt.subplots(ncols=2, figsize=(16, 5))
df["Risk"] = pd.Categorical(df["Risk"], categories=["good", "bad"])  # Convert the "Risk" column to a categorical variable
df["Risk"].value_counts().plot.pie(autopct="%.2f%%", colors=['#00FF7F', '#FF2424'], explode=(0.1, 0.1), ax=ax[0])
sns.countplot(x=df["Risk"], ax=ax[1], palette=['#00FF7F', '#FF2424'])

plt.show()


# In[93]:


# Display of bar graphs
fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(16, 20))
cat_list = ["Cat Age", "Sex", "Cat Job", "Housing", "Cat Duration", "Purpose"]
palette = ["red", "blue", "purple", "green", "yellow", "cyan"]
count = 0

for i in range(3):
    for j in range(2):
        sns.countplot(data=df, x=cat_list[count], ax=ax[i][j], palette=sns.dark_palette(palette[count], reverse=True))
        ax[i][j].set_xticklabels(ax[i][j].get_xticklabels(), rotation=30)
        count += 1

plt.tight_layout()
plt.show()


# In[94]:


# Encode the 'Sex' column as numeric values
df['Sex_Encoded'] = df['Sex'].map({'female': 0, 'male': 1})

# Display of bar graphs
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
sns.countplot(data=df, x='Sex_Encoded', ax=ax[0]).set_title('Male - Female Ratio')
sns.countplot(data=df, x='Risk', ax=ax[1]).set_title('Good - Bad Risk Ratio')

# Customize the x-axis labels on the first graph
ax[0].set_xticklabels(['Female', 'Male'])

plt.show()


# In[95]:


plt.figure(figsize=(16,5))
sns.countplot(x="Housing", hue="Risk", data=df).set_title("Housing and Frequency Graph by Risk", fontsize=15);
plt.show()


# In[96]:


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,5))
sns.countplot(x="Saving accounts", hue="Risk", data=df, ax=ax1);
sns.countplot(x="Checking account", hue="Risk", data=df, ax=ax2);
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
fig.show()


# In[97]:


plt.figure(figsize=(16,5))
sns.barplot(data=df,x="Sex",y="Age",hue="Cat Job",palette="hsv_r")


# In[98]:


plt.figure(figsize = (16, 5))
sns.stripplot(x = "Cat Age", y = "Credit amount", data = df)


# In[100]:


fig, ax = plt.subplots(2, 1, figsize=(16, 5))

sns.lineplot(data=df, x='Age', y='Credit amount', hue='Sex', lw=2, ax=ax[0]).set_title("Credit Amount Graph Depending on Age and Duration by Sex", fontsize=15)
sns.lineplot(data=df, x='Duration', y='Credit amount', hue='Sex', lw=2, ax=ax[1])

plt.tight_layout()  # Automatically adjust subplots

plt.show()


# In[111]:


# Style configuration
sns.set(style="whitegrid", palette="colorblind")

# Crear FacetGrid
grid = sns.FacetGrid(data=df, col="Risk", aspect=1.5, height=5)

# Plot points with custom colors
grid.map(sns.pointplot, "Cat Age", "Credit amount", "Sex", palette=["red", "green"])

# Customize titles and labels
grid.set_titles("{col_name}")
grid.set_xlabels("Cat Age")
grid.set_ylabels("Credit Amount")

# Show the graph
plt.show()


# In[113]:


# Filter the numeric columns of the DataFrame
numeric_columns = df.select_dtypes(include=[np.number])

# Create figure and heat map with numeric columns
plt.figure(figsize=(8.5, 5.5))
corr = sns.heatmap(numeric_columns.corr(), xticklabels=numeric_columns.columns, yticklabels=numeric_columns.columns, annot=True)

# Show the graph
plt.show()


# ## Data Preprocessing

# In[114]:


df.head()


# In[115]:


df["Age"],df["Duration"],df["Job"]=df["Cat Age"],df["Cat Duration"],df["Cat Job"]
df=df.drop(["Cat Age","Cat Duration","Cat Job"],axis=1)


# In[116]:


liste_columns=list(df.columns)
liste_columns.remove("Sex")
liste_columns.remove("Risk")
liste_columns.remove("Credit amount")
liste_columns.remove("Monthly pay")
liste_columns.remove("Credit amount^2")


# #### LabelEncoder

# In[117]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df["Sex"]=label.fit_transform(df["Sex"])
df["Risk"]=label.fit_transform(df["Risk"])
df=pd.get_dummies(df,columns=liste_columns,prefix=liste_columns)


# #### MinMaxScaler

# In[118]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df["Credit amount"]=scaler.fit_transform(df[["Credit amount"]])
df["Monthly pay"]=scaler.fit_transform(df[["Monthly pay"]])
df["Credit amount^2"]=scaler.fit_transform(df[["Credit amount^2"]])


# In[119]:


df.head()


# ### Building Models

# In[120]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score,auc,classification_report


# In[121]:


X=df.drop(["Risk"],axis=1)
Y=df["Risk"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# ### Decision Tree Model

# In[122]:


from sklearn.tree import DecisionTreeClassifier
cart_model=DecisionTreeClassifier(criterion='gini',max_depth=4,min_samples_leaf=54,min_samples_split=2).fit(X_train,Y_train)


# In[123]:


print("Train Accuracy Score : ",accuracy_score(Y_train,cart_model.predict(X_train)))
print("Test Accuracy Score : ",accuracy_score(Y_test,cart_model.predict(X_test)))


# In[124]:


print(classification_report(Y_test,cart_model.predict(X_test)))


# ### GradientBoosting Model

# In[128]:


from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier(learning_rate = 0.01,max_depth = 5,min_samples_split = 10,n_estimators = 100).fit(X_train, Y_train)


# In[129]:


print("Train Accuracy Score : ",accuracy_score(Y_train,gbm_model.predict(X_train)))
print("Test Accuracy Score : ",accuracy_score(Y_test,gbm_model.predict(X_test)))


# In[130]:


print(classification_report(Y_test,gbm_model.predict(X_test)))


# In[131]:


X_train.shape


# In[132]:


Importance=pd.DataFrame({"Values":gbm_model.feature_importances_*100},index=list(X_test.columns))
Importance.sort_values("Values",inplace=True,ascending=True)
Importance[28:].plot.barh()


# ### XGBoost Model

# In[143]:


get_ipython().system('pip install xgboost')


# In[135]:


from xgboost import XGBClassifier
xgb_model = XGBClassifier(learning_rate = 0.05, max_depth = 5,n_estimators=100,subsample=0.8).fit(X_train,Y_train)


# In[136]:


print("Train Accuracy Score : ",accuracy_score(Y_train,xgb_model.predict(X_train)))
print("Test Accuracy Score : ",accuracy_score(Y_test,xgb_model.predict(X_test)))


# In[137]:


print(classification_report(Y_test,xgb_model.predict(X_test)))


# In[138]:


Importance=pd.DataFrame({"Values":xgb_model.feature_importances_*100},index=list(X_test.columns))
Importance.sort_values("Values",inplace=True,ascending=True)
Importance[28:].plot.barh()


# ### LightGBM Model

# In[142]:


get_ipython().system('pip install lightgbm')


# In[ ]:


from lightgbm import LGBMClassifier
lgbm_model=LGBMClassifier(learning_rate=0.02,max_depth=3,min_child_samples=10,n_estimators=200,subsample=0.6).fit(X_train,Y_train)


# In[146]:


print("Train Accuracy Score : ",accuracy_score(Y_train,lgbm_model.predict(X_train)))
print("Test Accuracy Score : ",accuracy_score(Y_test,lgbm_model.predict(X_test)))


# In[147]:


print(classification_report(Y_test,lgbm_model.predict(X_test)))


# In[148]:


Importance=pd.DataFrame({"Values":lgbm_model.feature_importances_*100},index=list(X_test.columns))
Importance.sort_values("Values",inplace=True,ascending=True)
Importance[28:].plot.barh()


# In[152]:


from sklearn.metrics import roc_auc_score, roc_curve

list_model = [cart_model, gbm_model, xgb_model, lgbm_model]
list_model_name = ["DecisionTree Model", "GradientBoosting Model", "XGBoost Model", "LightGBM Model"]

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

for model, model_name, subplot in zip(list_model, list_model_name, ax.flatten()):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    logit_roc_auc = roc_auc_score(Y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred_proba)

    sns.lineplot(x=fpr, y=tpr, label="AUC = %0.2f" % logit_roc_auc, ax=subplot)
    subplot.plot([0, 1], [0, 1], color="red")
    subplot.legend(loc="lower right")
    subplot.set_title(model_name, fontsize=15)

fig.suptitle("ROC Curve", fontsize=18)
plt.tight_layout()
plt.show()


# In[154]:


from sklearn.metrics import accuracy_score

model_data = pd.DataFrame({
    "Model": ["DecisionTree Model", "GradientBoosting Model", "XGBoost Model", "LightGBM Model"],
    "Train Accuracy": [accuracy_score(Y_train, cart_model.predict(X_train)),
                       accuracy_score(Y_train, gbm_model.predict(X_train)),
                       accuracy_score(Y_train, xgb_model.predict(X_train)),
                       accuracy_score(Y_train, lgbm_model.predict(X_train))],
    "Test Accuracy": [accuracy_score(Y_test, cart_model.predict(X_test)),
                      accuracy_score(Y_test, gbm_model.predict(X_test)),
                      accuracy_score(Y_test, xgb_model.predict(X_test)),
                      accuracy_score(Y_test, lgbm_model.predict(X_test))]
})


# In[155]:


fig,ax=plt.subplots(ncols=2,figsize=(16,5))
sns.barplot(x="Model",y="Train Accuracy",data=model_data,ax=ax[0],palette="tab20c_r")
sns.barplot(x="Model",y="Test Accuracy",data=model_data,ax=ax[1],palette="tab20c_r")
ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=30)
ax[1].set_xticklabels(ax[0].get_xticklabels(),rotation=30);


# * We saw some important Features at the models results.
# * Now, we are creating a Tree image.
# * This Tree image shows us to what's going on the behind.

# In[156]:


df.head(1)


# * Selection 4 features according to importance

# In[157]:


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

variable = ["Risk","Monthly pay","Credit amount","Checking account_little","Checking account_moderate"]

data = df.loc[:,variable]

data.head(2)


# In[158]:


X = data.drop("Risk",axis=1)
y = data["Risk"]

forest = RandomForestClassifier(max_depth = 3, n_estimators=4)
forest.fit(X,y)


# In[160]:


estimator = forest.estimators_[3]


# In[161]:


target_names = ["0: good","1: bad"]


# In[162]:


from sklearn.tree import export_graphviz

export_graphviz(estimator,out_file="tree_limited.dot",feature_names=X.columns,
                class_names=target_names,rounded = True, proportion = False, precision = 2, filled = True)


# In[163]:


forest_1 = RandomForestClassifier(max_depth = None, n_estimators=4)
forest_1 = forest_1.fit(X,y)
estimator_non = forest_1.estimators_[3]


# In[170]:


get_ipython().system('pip install graphviz')


# In[180]:


import graphviz


# In[181]:


export_graphviz(estimator_non, out_file='tree_nonlimited.dot', feature_names = X.columns,
                class_names = target_names,
                rounded = True, proportion = False, precision = 2, filled = True)


# In[ ]:


get_ipython().system('dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600')


# In[184]:


from IPython.display import Image
Image(filename = 'media/tree_limited.png')

