# German Credit Risk Classification

## Content

##### The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes.

### The following steps were followed in this project:

1. Import Modules and Data
2. Data Analysis
3. Data Classification
4. Data Visualization
5. Data Preprocessing
6. Building Models
	* DecisionTree Model
	* GradientBoosting Model
	* XGBoost Model
	* LightGBM Model

* link_data = https://www.kaggle.com/kabure/german-credit-data-with-risk?select=german_credit_data.csv

## Import Modules and Data



```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
```


```python
df = pd.read_csv("data/german_credit_risk.csv")
df = df.iloc[:,1:]
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>little</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>moderate</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
      <td>bad</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>male</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>2096</td>
      <td>12</td>
      <td>education</td>
      <td>good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
      <td>good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
      <td>bad</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.dtypes
```




    Age                  int64
    Sex                 object
    Job                  int64
    Housing             object
    Saving accounts     object
    Checking account    object
    Credit amount        int64
    Duration             int64
    Purpose             object
    Risk                object
    dtype: object



## Variable Description

Meaning of the Values:

1. Age: Age of the person applying for the credit.
2. Sex: Gender of the person applying for the credit.
3. Job: 0,1,2,3 The values specified for the job in the form of 0,1,2,3.
4. Housing: own, rent or free.
5. Saving accounts: the amount of money in the person's bank account.
6. Checking account: cheque account.
7. Credit amount: Credit amount.
8. Duration: Time given for credit payment.
9. Purpose: Goal of credit application.
10. Risk:Credit application positive or negative.

## Data Analysis


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 10 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   Age               1000 non-null   int64 
     1   Sex               1000 non-null   object
     2   Job               1000 non-null   int64 
     3   Housing           1000 non-null   object
     4   Saving accounts   817 non-null    object
     5   Checking account  606 non-null    object
     6   Credit amount     1000 non-null   int64 
     7   Duration          1000 non-null   int64 
     8   Purpose           1000 non-null   object
     9   Risk              1000 non-null   object
    dtypes: int64(4), object(6)
    memory usage: 78.2+ KB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Job</th>
      <th>Credit amount</th>
      <th>Duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>35.546000</td>
      <td>1.904000</td>
      <td>3271.258000</td>
      <td>20.903000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.375469</td>
      <td>0.653614</td>
      <td>2822.736876</td>
      <td>12.058814</td>
    </tr>
    <tr>
      <th>min</th>
      <td>19.000000</td>
      <td>0.000000</td>
      <td>250.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>27.000000</td>
      <td>2.000000</td>
      <td>1365.500000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.000000</td>
      <td>2.000000</td>
      <td>2319.500000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>42.000000</td>
      <td>2.000000</td>
      <td>3972.250000</td>
      <td>24.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>75.000000</td>
      <td>3.000000</td>
      <td>18424.000000</td>
      <td>72.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Only numeric columns
numeric_columns = df.select_dtypes(include=[np.number])
correlation = numeric_columns.corr()
```


```python
print(correlation)
```

                        Age       Job  Credit amount  Duration
    Age            1.000000  0.015673       0.032716 -0.036136
    Job            0.015673  1.000000       0.285385  0.210910
    Credit amount  0.032716  0.285385       1.000000  0.624984
    Duration      -0.036136  0.210910       0.624984  1.000000
    


```python
df.dtypes
```




    Age                  int64
    Sex                 object
    Job                  int64
    Housing             object
    Saving accounts     object
    Checking account    object
    Credit amount        int64
    Duration             int64
    Purpose             object
    Risk                object
    dtype: object



### Unique Values


```python
def unique_value(data_set, column_name):
    return data_set[column_name].nunique()

print("Number of the Unique Values:")
print(unique_value(df,list(df.columns)))
```

    Number of the Unique Values:
    Age                  53
    Sex                   2
    Job                   4
    Housing               3
    Saving accounts       4
    Checking account      3
    Credit amount       921
    Duration             33
    Purpose               8
    Risk                  2
    dtype: int64
    

### Checking missing values 


```python
def missing_value_table(df):
    missing_value = df.isna().sum().sort_values(ascending=False)
    missing_value_percent = 100 * df.isna().sum()//len(df)
    missing_value_table = pd.concat([missing_value, missing_value_percent], axis=1)
    missing_value_table_return = missing_value_table.rename(columns = {0 : 'Missing Values', 1 : '% Value'})
    cm = sns.light_palette("lightgreen", as_cmap=True)
    missing_value_table_return = missing_value_table_return.style.background_gradient(cmap=cm)
    return missing_value_table_return
  
missing_value_table(df)
```




<style type="text/css">
#T_ae9be_row0_col0, #T_ae9be_row0_col1 {
  background-color: #90ee90;
  color: #000000;
}
#T_ae9be_row1_col0, #T_ae9be_row1_col1 {
  background-color: #c2f0c2;
  color: #000000;
}
#T_ae9be_row2_col0, #T_ae9be_row2_col1, #T_ae9be_row3_col0, #T_ae9be_row3_col1, #T_ae9be_row4_col0, #T_ae9be_row4_col1, #T_ae9be_row5_col0, #T_ae9be_row5_col1, #T_ae9be_row6_col0, #T_ae9be_row6_col1, #T_ae9be_row7_col0, #T_ae9be_row7_col1, #T_ae9be_row8_col0, #T_ae9be_row8_col1, #T_ae9be_row9_col0, #T_ae9be_row9_col1 {
  background-color: #edf2ed;
  color: #000000;
}
</style>
<table id="T_ae9be">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_ae9be_level0_col0" class="col_heading level0 col0" >Missing Values</th>
      <th id="T_ae9be_level0_col1" class="col_heading level0 col1" >% Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ae9be_level0_row0" class="row_heading level0 row0" >Checking account</th>
      <td id="T_ae9be_row0_col0" class="data row0 col0" >394</td>
      <td id="T_ae9be_row0_col1" class="data row0 col1" >39</td>
    </tr>
    <tr>
      <th id="T_ae9be_level0_row1" class="row_heading level0 row1" >Saving accounts</th>
      <td id="T_ae9be_row1_col0" class="data row1 col0" >183</td>
      <td id="T_ae9be_row1_col1" class="data row1 col1" >18</td>
    </tr>
    <tr>
      <th id="T_ae9be_level0_row2" class="row_heading level0 row2" >Age</th>
      <td id="T_ae9be_row2_col0" class="data row2 col0" >0</td>
      <td id="T_ae9be_row2_col1" class="data row2 col1" >0</td>
    </tr>
    <tr>
      <th id="T_ae9be_level0_row3" class="row_heading level0 row3" >Sex</th>
      <td id="T_ae9be_row3_col0" class="data row3 col0" >0</td>
      <td id="T_ae9be_row3_col1" class="data row3 col1" >0</td>
    </tr>
    <tr>
      <th id="T_ae9be_level0_row4" class="row_heading level0 row4" >Job</th>
      <td id="T_ae9be_row4_col0" class="data row4 col0" >0</td>
      <td id="T_ae9be_row4_col1" class="data row4 col1" >0</td>
    </tr>
    <tr>
      <th id="T_ae9be_level0_row5" class="row_heading level0 row5" >Housing</th>
      <td id="T_ae9be_row5_col0" class="data row5 col0" >0</td>
      <td id="T_ae9be_row5_col1" class="data row5 col1" >0</td>
    </tr>
    <tr>
      <th id="T_ae9be_level0_row6" class="row_heading level0 row6" >Credit amount</th>
      <td id="T_ae9be_row6_col0" class="data row6 col0" >0</td>
      <td id="T_ae9be_row6_col1" class="data row6 col1" >0</td>
    </tr>
    <tr>
      <th id="T_ae9be_level0_row7" class="row_heading level0 row7" >Duration</th>
      <td id="T_ae9be_row7_col0" class="data row7 col0" >0</td>
      <td id="T_ae9be_row7_col1" class="data row7 col1" >0</td>
    </tr>
    <tr>
      <th id="T_ae9be_level0_row8" class="row_heading level0 row8" >Purpose</th>
      <td id="T_ae9be_row8_col0" class="data row8 col0" >0</td>
      <td id="T_ae9be_row8_col1" class="data row8 col1" >0</td>
    </tr>
    <tr>
      <th id="T_ae9be_level0_row9" class="row_heading level0 row9" >Risk</th>
      <td id="T_ae9be_row9_col0" class="data row9 col0" >0</td>
      <td id="T_ae9be_row9_col1" class="data row9 col1" >0</td>
    </tr>
  </tbody>
</table>




### Sex Distribution


```python
pd.crosstab(df["Sex"],df["Risk"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Risk</th>
      <th>bad</th>
      <th>good</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>109</td>
      <td>201</td>
    </tr>
    <tr>
      <th>male</th>
      <td>191</td>
      <td>499</td>
    </tr>
  </tbody>
</table>
</div>



### Housing Distribution


```python
pd.crosstab(df["Housing"],df["Risk"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Risk</th>
      <th>bad</th>
      <th>good</th>
    </tr>
    <tr>
      <th>Housing</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>free</th>
      <td>44</td>
      <td>64</td>
    </tr>
    <tr>
      <th>own</th>
      <td>186</td>
      <td>527</td>
    </tr>
    <tr>
      <th>rent</th>
      <td>70</td>
      <td>109</td>
    </tr>
  </tbody>
</table>
</div>



* There is multiple groups in the "Purpose".
* At this situation we can apply ANOVA test.
* This way we will see the differences according to requisition of Credit Amount.


```python
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>little</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>moderate</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
      <td>bad</td>
    </tr>
  </tbody>
</table>
</div>




```python

from scipy import stats

df1 = df.copy()

df1 = df1[["Credit amount","Purpose"]]

group = pd.unique(df1.Purpose.values)

d_v1 = {grp:df1["Credit amount"][df1.Purpose == grp] for grp in group}
```

* One of conditions ANOVA test is equal variance.
* Applied levene and according to result, between groups variances are not equal.

### Applying levene


```python
stats.levene(d_v1['radio/TV'],d_v1['furniture/equipment'],d_v1['car'],d_v1['business'],d_v1['domestic appliances'],d_v1['repairs'],
                     d_v1['vacation/others'],d_v1['education'])
```




    LeveneResult(statistic=11.506286350981943, pvalue=4.177745359274538e-14)



* P value << 0.05


```python
f, p = stats.f_oneway(d_v1['radio/TV'],d_v1['furniture/equipment'],d_v1['car'],d_v1['business'],d_v1['domestic appliances'],d_v1['repairs'],
                     d_v1['vacation/others'],d_v1['education'])

("F statistics: "+str(f)+" | P value : "+str(p))
```




    'F statistics: 13.34142171179633 | P value : 1.585947764999813e-16'



* H0: There are no significant differences means of groups.

* H1: At least one group's mean is different.

* P value < 0.05

* Reject h0 hypothesis


```python
(df.groupby(by=["Purpose"])[["Credit amount"]].agg("sum") / df["Credit amount"].sum())*100
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Credit amount</th>
    </tr>
    <tr>
      <th>Purpose</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>business</th>
      <td>12.329507</td>
    </tr>
    <tr>
      <th>car</th>
      <td>38.819347</td>
    </tr>
    <tr>
      <th>domestic appliances</th>
      <td>0.549513</td>
    </tr>
    <tr>
      <th>education</th>
      <td>5.192895</td>
    </tr>
    <tr>
      <th>furniture/equipment</th>
      <td>16.969771</td>
    </tr>
    <tr>
      <th>radio/TV</th>
      <td>21.292818</td>
    </tr>
    <tr>
      <th>repairs</th>
      <td>1.834707</td>
    </tr>
    <tr>
      <th>vacation/others</th>
      <td>3.011441</td>
    </tr>
  </tbody>
</table>
</div>



* In the result, there is different between groups.
* In this query we can see difference

## Data Classification


```python
sns.set(font_scale=1,style="whitegrid")
fig,ax=plt.subplots(ncols=2,nrows=3,figsize=(16,12))
cat_list=["Age","Credit amount","Duration"]
count=0
for i in range(3):
    sns.distplot(df[cat_list[count]],ax=ax[i][0],kde=False,color="#F43EEC")
    sns.kdeplot(df[cat_list[count]],ax=ax[i][1],shade=True,color="#359F4B")
    count+=1
```


    
![png](output_38_0.png)
    


* "monthly pay" and "credit amount^2"(square) added in data frame.


```python
df["Monthly pay"] = (df["Credit amount"] / df["Duration"])
df["Credit amount^2"] = df["Credit amount"]**2
```

* 'Age' and 'Duration' columns Classification


```python
df.insert(1,"Cat Age",np.NaN)
df.loc[df["Age"]<25,"Cat Age"]="0-25"
df.loc[((df["Age"]>=25) & (df["Age"]<30)),"Cat Age"]="25-30"
df.loc[((df["Age"]>=30) & (df["Age"]<35)),"Cat Age"]="30-35"
df.loc[((df["Age"]>=35) & (df["Age"]<40)),"Cat Age"]="35-40"
df.loc[((df["Age"]>=40) & (df["Age"]<50)),"Cat Age"]="40-50"
df.loc[((df["Age"]>=50) & (df["Age"]<76)),"Cat Age"]="50-75"
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_128992\1021138275.py in ?()
    ----> 1 df.insert(1,"Cat Age",np.NaN)
          2 df.loc[df["Age"]<25,"Cat Age"]="0-25"
          3 df.loc[((df["Age"]>=25) & (df["Age"]<30)),"Cat Age"]="25-30"
          4 df.loc[((df["Age"]>=30) & (df["Age"]<35)),"Cat Age"]="30-35"
    

    E:\PYTHON ML PROJECTS\mlvenv\lib\site-packages\pandas\core\frame.py in ?(self, loc, column, value, allow_duplicates)
       4768                 "'self.flags.allows_duplicate_labels' is False."
       4769             )
       4770         if not allow_duplicates and column in self.columns:
       4771             # Should this be a different kind of error??
    -> 4772             raise ValueError(f"cannot insert {column}, already exists")
       4773         if not isinstance(loc, int):
       4774             raise TypeError("loc must be int")
       4775 
    

    ValueError: cannot insert Cat Age, already exists



```python
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
```


```python
df.insert(4,"Cat Job",df["Job"])
df["Cat Job"]=df["Cat Job"].astype("category")
df["Cat Job"]=df["Cat Job"].replace(0,"unskilled")
df["Cat Job"]=df["Cat Job"].replace(1,"resident")
df["Cat Job"]=df["Cat Job"].replace(2,"skilled")
df["Cat Job"]=df["Cat Job"].replace(3,"highly skilled")
```


```python
df["Job"]=pd.Categorical(df["Job"],categories=[0,1,2,3],ordered=True)
df["Cat Age"]=pd.Categorical(df["Cat Age"],categories=['0-25','25-30', '30-35','35-40','40-50','50-75'])
df["Cat Duration"]=pd.Categorical(df["Cat Duration"],categories=['0-12','12-24', '24-36','36-48','48-60','60-72'])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cat Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Cat Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Cat Duration</th>
      <th>Purpose</th>
      <th>Risk</th>
      <th>Monthly pay</th>
      <th>Credit amount^2</th>
      <th>Sex_Encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>50-75</td>
      <td>male</td>
      <td>2</td>
      <td>skilled</td>
      <td>own</td>
      <td>NaN</td>
      <td>little</td>
      <td>1169</td>
      <td>6</td>
      <td>0-12</td>
      <td>radio/TV</td>
      <td>good</td>
      <td>194.833333</td>
      <td>1366561</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>0-25</td>
      <td>female</td>
      <td>2</td>
      <td>skilled</td>
      <td>own</td>
      <td>little</td>
      <td>moderate</td>
      <td>5951</td>
      <td>48</td>
      <td>48-60</td>
      <td>radio/TV</td>
      <td>bad</td>
      <td>123.979167</td>
      <td>35414401</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>40-50</td>
      <td>male</td>
      <td>1</td>
      <td>resident</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>2096</td>
      <td>12</td>
      <td>12-24</td>
      <td>education</td>
      <td>good</td>
      <td>174.666667</td>
      <td>4393216</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>40-50</td>
      <td>male</td>
      <td>2</td>
      <td>skilled</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>7882</td>
      <td>42</td>
      <td>36-48</td>
      <td>furniture/equipment</td>
      <td>good</td>
      <td>187.666667</td>
      <td>62125924</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>50-75</td>
      <td>male</td>
      <td>2</td>
      <td>skilled</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>4870</td>
      <td>24</td>
      <td>24-36</td>
      <td>car</td>
      <td>bad</td>
      <td>202.916667</td>
      <td>23716900</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Data Visualization


```python
fig, ax = plt.subplots(ncols=2, figsize=(16, 5))
df["Risk"] = pd.Categorical(df["Risk"], categories=["good", "bad"])  # Convert the "Risk" column to a categorical variable
df["Risk"].value_counts().plot.pie(autopct="%.2f%%", colors=['#00FF7F', '#FF2424'], explode=(0.1, 0.1), ax=ax[0])
sns.countplot(x=df["Risk"], ax=ax[1], palette=['#00FF7F', '#FF2424'])

plt.show()
```


    
![png](output_48_0.png)
    



```python
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
```


    
![png](output_49_0.png)
    



```python
# Encode the 'Sex' column as numeric values
df['Sex_Encoded'] = df['Sex'].map({'female': 0, 'male': 1})

# Display of bar graphs
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
sns.countplot(data=df, x='Sex_Encoded', ax=ax[0]).set_title('Male - Female Ratio')
sns.countplot(data=df, x='Risk', ax=ax[1]).set_title('Good - Bad Risk Ratio')

# Customize the x-axis labels on the first graph
ax[0].set_xticklabels(['Female', 'Male'])

plt.show()
```


    
![png](output_50_0.png)
    



```python
plt.figure(figsize=(16,5))
sns.countplot(x="Housing", hue="Risk", data=df).set_title("Housing and Frequency Graph by Risk", fontsize=15);
plt.show()
```


    
![png](output_51_0.png)
    



```python
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,5))
sns.countplot(x="Saving accounts", hue="Risk", data=df, ax=ax1);
sns.countplot(x="Checking account", hue="Risk", data=df, ax=ax2);
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
fig.show()
```


    
![png](output_52_0.png)
    



```python
plt.figure(figsize=(16,5))
sns.barplot(data=df,x="Sex",y="Age",hue="Cat Job",palette="hsv_r")
```




    <Axes: xlabel='Sex', ylabel='Age'>




    
![png](output_53_1.png)
    



```python
plt.figure(figsize = (16, 5))
sns.stripplot(x = "Cat Age", y = "Credit amount", data = df)
```




    <Axes: xlabel='Cat Age', ylabel='Credit amount'>




    
![png](output_54_1.png)
    



```python
fig, ax = plt.subplots(2, 1, figsize=(16, 5))

sns.lineplot(data=df, x='Age', y='Credit amount', hue='Sex', lw=2, ax=ax[0]).set_title("Credit Amount Graph Depending on Age and Duration by Sex", fontsize=15)
sns.lineplot(data=df, x='Duration', y='Credit amount', hue='Sex', lw=2, ax=ax[1])

plt.tight_layout()  # Automatically adjust subplots

plt.show()
```


    
![png](output_55_0.png)
    



```python
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
```


    
![png](output_56_0.png)
    



```python
# Filter the numeric columns of the DataFrame
numeric_columns = df.select_dtypes(include=[np.number])

# Create figure and heat map with numeric columns
plt.figure(figsize=(8.5, 5.5))
corr = sns.heatmap(numeric_columns.corr(), xticklabels=numeric_columns.columns, yticklabels=numeric_columns.columns, annot=True)

# Show the graph
plt.show()
```


    
![png](output_57_0.png)
    


## Data Preprocessing


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cat Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Cat Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Cat Duration</th>
      <th>Purpose</th>
      <th>Risk</th>
      <th>Monthly pay</th>
      <th>Credit amount^2</th>
      <th>Sex_Encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>50-75</td>
      <td>male</td>
      <td>2</td>
      <td>skilled</td>
      <td>own</td>
      <td>NaN</td>
      <td>little</td>
      <td>1169</td>
      <td>6</td>
      <td>0-12</td>
      <td>radio/TV</td>
      <td>good</td>
      <td>194.833333</td>
      <td>1366561</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>0-25</td>
      <td>female</td>
      <td>2</td>
      <td>skilled</td>
      <td>own</td>
      <td>little</td>
      <td>moderate</td>
      <td>5951</td>
      <td>48</td>
      <td>48-60</td>
      <td>radio/TV</td>
      <td>bad</td>
      <td>123.979167</td>
      <td>35414401</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49</td>
      <td>40-50</td>
      <td>male</td>
      <td>1</td>
      <td>resident</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>2096</td>
      <td>12</td>
      <td>12-24</td>
      <td>education</td>
      <td>good</td>
      <td>174.666667</td>
      <td>4393216</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>40-50</td>
      <td>male</td>
      <td>2</td>
      <td>skilled</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>7882</td>
      <td>42</td>
      <td>36-48</td>
      <td>furniture/equipment</td>
      <td>good</td>
      <td>187.666667</td>
      <td>62125924</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
      <td>50-75</td>
      <td>male</td>
      <td>2</td>
      <td>skilled</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>4870</td>
      <td>24</td>
      <td>24-36</td>
      <td>car</td>
      <td>bad</td>
      <td>202.916667</td>
      <td>23716900</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["Age"],df["Duration"],df["Job"]=df["Cat Age"],df["Cat Duration"],df["Cat Job"]
df=df.drop(["Cat Age","Cat Duration","Cat Job"],axis=1)
```


```python
liste_columns=list(df.columns)
liste_columns.remove("Sex")
liste_columns.remove("Risk")
liste_columns.remove("Credit amount")
liste_columns.remove("Monthly pay")
liste_columns.remove("Credit amount^2")
```

#### LabelEncoder


```python
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df["Sex"]=label.fit_transform(df["Sex"])
df["Risk"]=label.fit_transform(df["Risk"])
df=pd.get_dummies(df,columns=liste_columns,prefix=liste_columns)
```

#### MinMaxScaler


```python
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df["Credit amount"]=scaler.fit_transform(df[["Credit amount"]])
df["Monthly pay"]=scaler.fit_transform(df[["Monthly pay"]])
df["Credit amount^2"]=scaler.fit_transform(df[["Credit amount^2"]])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Credit amount</th>
      <th>Risk</th>
      <th>Monthly pay</th>
      <th>Credit amount^2</th>
      <th>Age_0-25</th>
      <th>Age_25-30</th>
      <th>Age_30-35</th>
      <th>Age_35-40</th>
      <th>Age_40-50</th>
      <th>...</th>
      <th>Purpose_business</th>
      <th>Purpose_car</th>
      <th>Purpose_domestic appliances</th>
      <th>Purpose_education</th>
      <th>Purpose_furniture/equipment</th>
      <th>Purpose_radio/TV</th>
      <th>Purpose_repairs</th>
      <th>Purpose_vacation/others</th>
      <th>Sex_Encoded_0</th>
      <th>Sex_Encoded_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.050567</td>
      <td>1</td>
      <td>0.069461</td>
      <td>0.003842</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.313690</td>
      <td>0</td>
      <td>0.040642</td>
      <td>0.104166</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.101574</td>
      <td>1</td>
      <td>0.061259</td>
      <td>0.012761</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.419941</td>
      <td>1</td>
      <td>0.066546</td>
      <td>0.182872</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.254209</td>
      <td>0</td>
      <td>0.072749</td>
      <td>0.069699</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>



### Building Models


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score,auc,classification_report
```


```python
X=df.drop(["Risk"],axis=1)
Y=df["Risk"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
```

### Decision Tree Model


```python
from sklearn.tree import DecisionTreeClassifier
cart_model=DecisionTreeClassifier(criterion='gini',max_depth=4,min_samples_leaf=54,min_samples_split=2).fit(X_train,Y_train)
```


```python
print("Train Accuracy Score : ",accuracy_score(Y_train,cart_model.predict(X_train)))
print("Test Accuracy Score : ",accuracy_score(Y_test,cart_model.predict(X_test)))
```

    Train Accuracy Score :  0.7571428571428571
    Test Accuracy Score :  0.71
    


```python
print(classification_report(Y_test,cart_model.predict(X_test)))
```

                  precision    recall  f1-score   support
    
               0       0.49      0.44      0.47        86
               1       0.78      0.82      0.80       214
    
        accuracy                           0.71       300
       macro avg       0.64      0.63      0.63       300
    weighted avg       0.70      0.71      0.70       300
    
    

### GradientBoosting Model


```python
from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier(learning_rate = 0.01,max_depth = 5,min_samples_split = 10,n_estimators = 100).fit(X_train, Y_train)
```


```python
print("Train Accuracy Score : ",accuracy_score(Y_train,gbm_model.predict(X_train)))
print("Test Accuracy Score : ",accuracy_score(Y_test,gbm_model.predict(X_test)))
```

    Train Accuracy Score :  0.8385714285714285
    Test Accuracy Score :  0.74
    


```python
print(classification_report(Y_test,gbm_model.predict(X_test)))
```

                  precision    recall  f1-score   support
    
               0       0.65      0.20      0.30        86
               1       0.75      0.96      0.84       214
    
        accuracy                           0.74       300
       macro avg       0.70      0.58      0.57       300
    weighted avg       0.72      0.74      0.69       300
    
    


```python
X_train.shape
```




    (700, 40)




```python
Importance=pd.DataFrame({"Values":gbm_model.feature_importances_*100},index=list(X_test.columns))
Importance.sort_values("Values",inplace=True,ascending=True)
Importance[28:].plot.barh()
```




    <Axes: >




    
![png](output_79_1.png)
    


### XGBoost Model


```python
!pip install xgboost
```

    Requirement already satisfied: xgboost in e:\python ml projects\mlvenv\lib\site-packages (1.7.6)
    Requirement already satisfied: numpy in e:\python ml projects\mlvenv\lib\site-packages (from xgboost) (1.25.1)
    Requirement already satisfied: scipy in e:\python ml projects\mlvenv\lib\site-packages (from xgboost) (1.11.1)
    


```python
from xgboost import XGBClassifier
xgb_model = XGBClassifier(learning_rate = 0.05, max_depth = 5,n_estimators=100,subsample=0.8).fit(X_train,Y_train)
```


```python
print("Train Accuracy Score : ",accuracy_score(Y_train,xgb_model.predict(X_train)))
print("Test Accuracy Score : ",accuracy_score(Y_test,xgb_model.predict(X_test)))
```

    Train Accuracy Score :  0.9014285714285715
    Test Accuracy Score :  0.7633333333333333
    


```python
print(classification_report(Y_test,xgb_model.predict(X_test)))
```

                  precision    recall  f1-score   support
    
               0       0.63      0.42      0.50        86
               1       0.79      0.90      0.84       214
    
        accuracy                           0.76       300
       macro avg       0.71      0.66      0.67       300
    weighted avg       0.75      0.76      0.75       300
    
    


```python
Importance=pd.DataFrame({"Values":xgb_model.feature_importances_*100},index=list(X_test.columns))
Importance.sort_values("Values",inplace=True,ascending=True)
Importance[28:].plot.barh()
```




    <Axes: >




    
![png](output_85_1.png)
    


### LightGBM Model


```python
!pip install lightgbm
```

    Requirement already satisfied: lightgbm in e:\python ml projects\mlvenv\lib\site-packages (4.0.0)
    Requirement already satisfied: numpy in e:\python ml projects\mlvenv\lib\site-packages (from lightgbm) (1.25.1)
    Requirement already satisfied: scipy in e:\python ml projects\mlvenv\lib\site-packages (from lightgbm) (1.11.1)
    


```python
from lightgbm import LGBMClassifier
lgbm_model=LGBMClassifier(learning_rate=0.02,max_depth=3,min_child_samples=10,n_estimators=200,subsample=0.6).fit(X_train,Y_train)
```


```python
print("Train Accuracy Score : ",accuracy_score(Y_train,lgbm_model.predict(X_train)))
print("Test Accuracy Score : ",accuracy_score(Y_test,lgbm_model.predict(X_test)))
```

    Train Accuracy Score :  0.8157142857142857
    Test Accuracy Score :  0.7366666666666667
    


```python
print(classification_report(Y_test,lgbm_model.predict(X_test)))
```

                  precision    recall  f1-score   support
    
               0       0.59      0.28      0.38        86
               1       0.76      0.92      0.83       214
    
        accuracy                           0.74       300
       macro avg       0.67      0.60      0.61       300
    weighted avg       0.71      0.74      0.70       300
    
    


```python
Importance=pd.DataFrame({"Values":lgbm_model.feature_importances_*100},index=list(X_test.columns))
Importance.sort_values("Values",inplace=True,ascending=True)
Importance[28:].plot.barh()
```




    <Axes: >




    
![png](output_91_1.png)
    



```python
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
```


    
![png](output_92_0.png)
    



```python
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
```


```python
fig,ax=plt.subplots(ncols=2,figsize=(16,5))
sns.barplot(x="Model",y="Train Accuracy",data=model_data,ax=ax[0],palette="tab20c_r")
sns.barplot(x="Model",y="Test Accuracy",data=model_data,ax=ax[1],palette="tab20c_r")
ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=30)
ax[1].set_xticklabels(ax[0].get_xticklabels(),rotation=30);
```


    
![png](output_94_0.png)
    


* We saw some important Features at the models results.
* Now, we are creating a Tree image.
* This Tree image shows us to what's going on the behind.


```python
df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Credit amount</th>
      <th>Risk</th>
      <th>Monthly pay</th>
      <th>Credit amount^2</th>
      <th>Age_0-25</th>
      <th>Age_25-30</th>
      <th>Age_30-35</th>
      <th>Age_35-40</th>
      <th>Age_40-50</th>
      <th>...</th>
      <th>Purpose_business</th>
      <th>Purpose_car</th>
      <th>Purpose_domestic appliances</th>
      <th>Purpose_education</th>
      <th>Purpose_furniture/equipment</th>
      <th>Purpose_radio/TV</th>
      <th>Purpose_repairs</th>
      <th>Purpose_vacation/others</th>
      <th>Sex_Encoded_0</th>
      <th>Sex_Encoded_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.050567</td>
      <td>1</td>
      <td>0.069461</td>
      <td>0.003842</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 41 columns</p>
</div>



* Selection 4 features according to importance


```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

variable = ["Risk","Monthly pay","Credit amount","Checking account_little","Checking account_moderate"]

data = df.loc[:,variable]

data.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Risk</th>
      <th>Monthly pay</th>
      <th>Credit amount</th>
      <th>Checking account_little</th>
      <th>Checking account_moderate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.069461</td>
      <td>0.050567</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.040642</td>
      <td>0.313690</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = data.drop("Risk",axis=1)
y = data["Risk"]

forest = RandomForestClassifier(max_depth = 3, n_estimators=4)
forest.fit(X,y)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(max_depth=3, n_estimators=4)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(max_depth=3, n_estimators=4)</pre></div></div></div></div></div>




```python
estimator = forest.estimators_[3]
```


```python
target_names = ["0: good","1: bad"]
```


```python
from sklearn.tree import export_graphviz

export_graphviz(estimator,out_file="tree_limited.dot",feature_names=X.columns,
                class_names=target_names,rounded = True, proportion = False, precision = 2, filled = True)
```


```python
forest_1 = RandomForestClassifier(max_depth = None, n_estimators=4)
forest_1 = forest_1.fit(X,y)
estimator_non = forest_1.estimators_[3]
```


```python
!pip install graphviz
```

    Collecting graphviz
      Downloading graphviz-0.20.1-py3-none-any.whl (47 kB)
         ---------------------------------------- 0.0/47.0 kB ? eta -:--:--
         ---------------------------------------- 47.0/47.0 kB 1.2 MB/s eta 0:00:00
    Installing collected packages: graphviz
    Successfully installed graphviz-0.20.1
    


```python
import graphviz
```


```python
export_graphviz(estimator_non, out_file='tree_nonlimited.dot', feature_names = X.columns,
                class_names = target_names,
                rounded = True, proportion = False, precision = 2, filled = True)
```


```python
!dot -Tpng tree_limited.dot -o tree_limited.png -Gdpi=600
```


```python
from IPython.display import Image
Image(filename = 'media/tree_limited.png')
```




    
![png](output_108_0.png)
    


