<center><h1><b><font size="6">Python Machine Learning Financial Analysis</font></b></h1></center>

#### Import Libraries


```python
import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
```

#### Load the data


```python
data = pd.read_excel('./data/default_of_credit_card_clients.xls', header=0)
data.head()
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
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>120000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>50000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>50000</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



#### Information about the different columns:



**ID**: Account ID

**LIMIT_BAL**: Credit limit (in New Taiwanese dollars (NT)) including both individual and family (supplementary) credit

**SEX**: 1 = male; 2 = female

**EDUCATION**: 1 = graduate school; 2 = university; 3 = high school; 4 = others

**MARRIAGE**: Marital status (1 = married; 2 = single; 3 = others)

**AGE**: Age

**PAY_1 - PAY_6**: History of past payments. History from April to September. The rating scale is as follows: -2 = No consumption; -1 = Paid in full; 0 = The use of revolving credit; 1 = Payment delay for one month; 2 = Payment delay for two months; and so on up to 9 = Payment delay for nine months and above

**BILL_AMT1 - BILL_AMT6**: Amount of bill statement. BILL_AMT1 represents the amount of the bill statement in September, and 
BILL_AMT6 represents the amount of the bill statement in April.

**PAY_AMT1 - PAY_AMT6**: Amount of previous payment (in NT dollars).


```python
data.shape
```




    (30000, 25)




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30000 entries, 0 to 29999
    Data columns (total 25 columns):
     #   Column                      Non-Null Count  Dtype
    ---  ------                      --------------  -----
     0   ID                          30000 non-null  int64
     1   LIMIT_BAL                   30000 non-null  int64
     2   SEX                         30000 non-null  int64
     3   EDUCATION                   30000 non-null  int64
     4   MARRIAGE                    30000 non-null  int64
     5   AGE                         30000 non-null  int64
     6   PAY_1                       30000 non-null  int64
     7   PAY_2                       30000 non-null  int64
     8   PAY_3                       30000 non-null  int64
     9   PAY_4                       30000 non-null  int64
     10  PAY_5                       30000 non-null  int64
     11  PAY_6                       30000 non-null  int64
     12  BILL_AMT1                   30000 non-null  int64
     13  BILL_AMT2                   30000 non-null  int64
     14  BILL_AMT3                   30000 non-null  int64
     15  BILL_AMT4                   30000 non-null  int64
     16  BILL_AMT5                   30000 non-null  int64
     17  BILL_AMT6                   30000 non-null  int64
     18  PAY_AMT1                    30000 non-null  int64
     19  PAY_AMT2                    30000 non-null  int64
     20  PAY_AMT3                    30000 non-null  int64
     21  PAY_AMT4                    30000 non-null  int64
     22  PAY_AMT5                    30000 non-null  int64
     23  PAY_AMT6                    30000 non-null  int64
     24  default payment next month  30000 non-null  int64
    dtypes: int64(25)
    memory usage: 5.7 MB
    

##### unique ID Values in the dataset


```python
data['ID'].nunique()
```




    30000




```python
data['ID'].value_counts()
```




    ID
    1        1
    19997    1
    20009    1
    20008    1
    20007    1
            ..
    9996     1
    9995     1
    9994     1
    9993     1
    30000    1
    Name: count, Length: 30000, dtype: int64




```python
id_counts =data['ID'].value_counts()
id_counts[:3] # the first 3 
```




    ID
    1        1
    19997    1
    20009    1
    Name: count, dtype: int64




```python
id_counts.value_counts()
```




    count
    1    30000
    Name: count, dtype: int64




```python
# boolean mask
bool_mask = id_counts == 2
bool_mask[:5]
```




    ID
    1        False
    19997    False
    20009    False
    20008    False
    20007    False
    Name: count, dtype: bool



#### Exploring financial history


```python
data = pd.read_excel('./data/default_of_credit_card_clients.xls', header=0)
data.head()
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
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default payment next month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>120000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>50000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>50000</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>




```python
data.shape
```




    (30000, 25)




```python
data.columns
```




    Index(['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1',
           'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
           'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
           'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
           'default payment next month'],
          dtype='object')



#### PAY Columns


```python
pay_columns = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
data[pay_columns].describe()
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
      <th>PAY_1</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.016700</td>
      <td>-0.133767</td>
      <td>-0.166200</td>
      <td>-0.220667</td>
      <td>-0.266200</td>
      <td>-0.291100</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.123802</td>
      <td>1.197186</td>
      <td>1.196868</td>
      <td>1.169139</td>
      <td>1.133187</td>
      <td>1.149988</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### allows you to see payment behavior, where most of the data and payment trends of credit card customers are concentrated. In this we know the maximum and minimum values, and it is observed that most of the customers are up to date with their payments since it reflects the highest concentration of data.


```python
data[pay_columns[0]].value_counts().sort_index()
```




    PAY_1
    -2     2759
    -1     5686
     0    14737
     1     3688
     2     2667
     3      322
     4       76
     5       26
     6       11
     7        9
     8       19
    Name: count, dtype: int64



##### Grafic


```python
data[pay_columns[0]].hist()
```




    <Axes: >




    
![png](output_23_1.png)
    


##### 0 = represents loan payments 
##### 1 = represents non-payments, overdue



```python
data['default payment next month'].value_counts()     
```




    default payment next month
    0    23364
    1     6636
    Name: count, dtype: int64



#### Every PAY column grafic

*we can see that from PAY_2 the information is not correct*.



```python
import matplotlib as mlp
mlp.rcParams['figure.dpi'] = 400 # high definition
mlp.rcParams['font.size'] = 4 # text size
data[pay_columns].hist(layout=(2, 3))
```




    array([[<Axes: title={'center': 'PAY_1'}>,
            <Axes: title={'center': 'PAY_2'}>,
            <Axes: title={'center': 'PAY_3'}>],
           [<Axes: title={'center': 'PAY_4'}>,
            <Axes: title={'center': 'PAY_5'}>,
            <Axes: title={'center': 'PAY_6'}>]], dtype=object)




    
![png](output_27_1.png)
    



```python
data.loc[data['PAY_2'] == 2, ['PAY_2', 'PAY_3']].head()
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
      <th>PAY_2</th>
      <th>PAY_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



##### the others columns


```python
bill_feats = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
pay_amt_feats = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
```


```python
data[bill_feats].describe()
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
      <th>BILL_AMT1</th>
      <th>BILL_AMT2</th>
      <th>BILL_AMT3</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>3.000000e+04</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
      <td>30000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>51223.330900</td>
      <td>49179.075167</td>
      <td>4.701315e+04</td>
      <td>43262.948967</td>
      <td>40311.400967</td>
      <td>38871.760400</td>
    </tr>
    <tr>
      <th>std</th>
      <td>73635.860576</td>
      <td>71173.768783</td>
      <td>6.934939e+04</td>
      <td>64332.856134</td>
      <td>60797.155770</td>
      <td>59554.107537</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-165580.000000</td>
      <td>-69777.000000</td>
      <td>-1.572640e+05</td>
      <td>-170000.000000</td>
      <td>-81334.000000</td>
      <td>-339603.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3558.750000</td>
      <td>2984.750000</td>
      <td>2.666250e+03</td>
      <td>2326.750000</td>
      <td>1763.000000</td>
      <td>1256.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22381.500000</td>
      <td>21200.000000</td>
      <td>2.008850e+04</td>
      <td>19052.000000</td>
      <td>18104.500000</td>
      <td>17071.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>67091.000000</td>
      <td>64006.250000</td>
      <td>6.016475e+04</td>
      <td>54506.000000</td>
      <td>50190.500000</td>
      <td>49198.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>964511.000000</td>
      <td>983931.000000</td>
      <td>1.664089e+06</td>
      <td>891586.000000</td>
      <td>927171.000000</td>
      <td>961664.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### customer receipts


```python
data[bill_feats].hist(layout=(2, 3))
```




    array([[<Axes: title={'center': 'BILL_AMT1'}>,
            <Axes: title={'center': 'BILL_AMT2'}>,
            <Axes: title={'center': 'BILL_AMT3'}>],
           [<Axes: title={'center': 'BILL_AMT4'}>,
            <Axes: title={'center': 'BILL_AMT5'}>,
            <Axes: title={'center': 'BILL_AMT6'}>]], dtype=object)




    
![png](output_33_1.png)
    


#### customer payments


```python
data[pay_amt_feats].hist(layout=(2, 3))
```




    array([[<Axes: title={'center': 'PAY_AMT1'}>,
            <Axes: title={'center': 'PAY_AMT2'}>,
            <Axes: title={'center': 'PAY_AMT3'}>],
           [<Axes: title={'center': 'PAY_AMT4'}>,
            <Axes: title={'center': 'PAY_AMT5'}>,
            <Axes: title={'center': 'PAY_AMT6'}>]], dtype=object)




    
![png](output_35_1.png)
    



```python
pay_zero_mask = data[pay_amt_feats] == 0
pay_zero_mask.sum()
```




    PAY_AMT1    5249
    PAY_AMT2    5396
    PAY_AMT3    5968
    PAY_AMT4    6408
    PAY_AMT5    6703
    PAY_AMT6    7173
    dtype: int64



##### graphical payments representation without non-paying customers


```python
data[pay_amt_feats][~pay_zero_mask].apply(np.log10).hist(layout=(2, 3))
```




    array([[<Axes: title={'center': 'PAY_AMT1'}>,
            <Axes: title={'center': 'PAY_AMT2'}>,
            <Axes: title={'center': 'PAY_AMT3'}>],
           [<Axes: title={'center': 'PAY_AMT4'}>,
            <Axes: title={'center': 'PAY_AMT5'}>,
            <Axes: title={'center': 'PAY_AMT6'}>]], dtype=object)




    
![png](output_38_1.png)
    


#### credit limit column


```python
data['LIMIT_BAL'].hist()
```




    <Axes: >




    
![png](output_40_1.png)
    


#### Machine Learning


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data[['EDUCATION', 'PAY_1', 'LIMIT_BAL']].values, #columns to use / independent
                                                    data['default payment next month'], #variable to predict / dependent
                                                    test_size=.2, random_state=24) # size 20% - of the data to testing - X_test + y_test / 80% X_train
# size 20% - of the data to testing - X_test + y_test / 80% X_train
# 80 % (X_train) of EDUCATION, PAY_1 & LIMIT_BAL
# 20 % (X_test) of EDUCATION, PAY_1 & LIMIT_BAL
# 80 % (y_train) of default payment next month column (to predict)
# 20 % (y_test) of default payment next month column (to predict)
```


```python
# 24000 rows to train the model
# 6000 rows to test the model

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (24000, 3)
    (6000, 3)
    (24000,)
    (6000,)
    

##### 0 = 23364 / paying customers

##### 1 = 6636 / non-paying customers

there is no symmetry in the data, we must find an algorithm that adapts to this business need.


```python
data['default payment next month'].value_counts()
```




    default payment next month
    0    23364
    1     6636
    Name: count, dtype: int64



#### RandomForestClassifier

This algorithm can handle scenarios well when there are large differences in the data or no symmetry, e.g. payments and non-payments.


```python
from sklearn.ensemble import RandomForestClassifier

bosque_aleatorio = RandomForestClassifier(n_estimators=100, random_state=24) #n_estimators = 100 = numbers of forest (the higher the number, the more accurate it can be, but the harder it is to train)

bosque_aleatorio.fit(X_train, y_train) #fit the model = values to get and train...

y_pred = bosque_aleatorio.predict(X_test) #variable to predict / prediction of values
```

#### show the predicted values and compare it with the values of the y_test


```python
y_pred[:15]
```




    array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=int64)



#### show the real values and compare it with the values of the y_pred


```python
y_test[:15]
```




    25706    0
    2412     1
    13346    1
    23685    0
    9943     0
    28908    1
    7167     0
    25657    0
    16167    0
    3283     0
    13379    0
    10724    0
    12366    0
    5223     1
    8781     1
    Name: default payment next month, dtype: int64



##### Accuracy / ML / RandomForestClassifierModel

81.15 % of the data. So we can say that the model is quite good since it used more than 80% of the data. However, we can add other columns of interest to the model and it can affect the accuracy of the model as well.


```python
from sklearn.metrics import accuracy_score

accury = accuracy_score(y_test, y_pred)
accury
```




    0.8115



##### LogisticRegression


```python
from sklearn import linear_model

logit_model = linear_model.LogisticRegression()
logit_model.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
y_pred = logit_model.predict(X_test)
```


```python
y_pred[:15]
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int64)




```python
y_test[:15]
```




    25706    0
    2412     1
    13346    1
    23685    0
    9943     0
    28908    1
    7167     0
    25657    0
    16167    0
    3283     0
    13379    0
    10724    0
    12366    0
    5223     1
    8781     1
    Name: default payment next month, dtype: int64



##### Accuracy / ML / LogiticRegression


```python
accury = accuracy_score(y_test, y_pred)
accury
```




    0.7738333333333334



##### Conclusion

The Logistic **Regression model (77.38% - accuracy)** is not very good for this data scenario unlike the **RandomForestClassifier model (81.15% - accuracy)**. 
