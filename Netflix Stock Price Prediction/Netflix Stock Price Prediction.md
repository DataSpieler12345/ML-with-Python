# Netflix Stock Price Prediction

### Import Libraries


```python
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
```

### Data Loading


```python
df = pd.read_csv('data/NFLX.csv')
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
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-02-05</td>
      <td>262.000000</td>
      <td>267.899994</td>
      <td>250.029999</td>
      <td>254.259995</td>
      <td>254.259995</td>
      <td>11896100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-02-06</td>
      <td>247.699997</td>
      <td>266.700012</td>
      <td>245.000000</td>
      <td>265.720001</td>
      <td>265.720001</td>
      <td>12595800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-02-07</td>
      <td>266.579987</td>
      <td>272.450012</td>
      <td>264.329987</td>
      <td>264.559998</td>
      <td>264.559998</td>
      <td>8981500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-02-08</td>
      <td>267.079987</td>
      <td>267.619995</td>
      <td>250.000000</td>
      <td>250.100006</td>
      <td>250.100006</td>
      <td>9306700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-02-09</td>
      <td>253.850006</td>
      <td>255.800003</td>
      <td>236.110001</td>
      <td>249.470001</td>
      <td>249.470001</td>
      <td>16906900</td>
    </tr>
  </tbody>
</table>
</div>




```python
#DF data Viz
viz = df.copy()
viz.head()
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
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-02-05</td>
      <td>262.000000</td>
      <td>267.899994</td>
      <td>250.029999</td>
      <td>254.259995</td>
      <td>254.259995</td>
      <td>11896100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-02-06</td>
      <td>247.699997</td>
      <td>266.700012</td>
      <td>245.000000</td>
      <td>265.720001</td>
      <td>265.720001</td>
      <td>12595800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-02-07</td>
      <td>266.579987</td>
      <td>272.450012</td>
      <td>264.329987</td>
      <td>264.559998</td>
      <td>264.559998</td>
      <td>8981500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-02-08</td>
      <td>267.079987</td>
      <td>267.619995</td>
      <td>250.000000</td>
      <td>250.100006</td>
      <td>250.100006</td>
      <td>9306700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-02-09</td>
      <td>253.850006</td>
      <td>255.800003</td>
      <td>236.110001</td>
      <td>249.470001</td>
      <td>249.470001</td>
      <td>16906900</td>
    </tr>
  </tbody>
</table>
</div>



### Data Preparation


```python
df.isnull().sum()
```




    Date         0
    Open         0
    High         0
    Low          0
    Close        0
    Adj Close    0
    Volume       0
    dtype: int64




```python
df.shape
```




    (1009, 7)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1009 entries, 0 to 1008
    Data columns (total 7 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Date       1009 non-null   object 
     1   Open       1009 non-null   float64
     2   High       1009 non-null   float64
     3   Low        1009 non-null   float64
     4   Close      1009 non-null   float64
     5   Adj Close  1009 non-null   float64
     6   Volume     1009 non-null   int64  
    dtypes: float64(5), int64(1), object(1)
    memory usage: 55.3+ KB
    


```python
df.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Open</th>
      <td>1009.0</td>
      <td>4.190597e+02</td>
      <td>1.085375e+02</td>
      <td>2.339200e+02</td>
      <td>3.314900e+02</td>
      <td>3.777700e+02</td>
      <td>5.091300e+02</td>
      <td>6.923500e+02</td>
    </tr>
    <tr>
      <th>High</th>
      <td>1009.0</td>
      <td>4.253207e+02</td>
      <td>1.092630e+02</td>
      <td>2.506500e+02</td>
      <td>3.363000e+02</td>
      <td>3.830100e+02</td>
      <td>5.156300e+02</td>
      <td>7.009900e+02</td>
    </tr>
    <tr>
      <th>Low</th>
      <td>1009.0</td>
      <td>4.123740e+02</td>
      <td>1.075559e+02</td>
      <td>2.312300e+02</td>
      <td>3.260000e+02</td>
      <td>3.708800e+02</td>
      <td>5.025300e+02</td>
      <td>6.860900e+02</td>
    </tr>
    <tr>
      <th>Close</th>
      <td>1009.0</td>
      <td>4.190007e+02</td>
      <td>1.082900e+02</td>
      <td>2.338800e+02</td>
      <td>3.316200e+02</td>
      <td>3.786700e+02</td>
      <td>5.090800e+02</td>
      <td>6.916900e+02</td>
    </tr>
    <tr>
      <th>Adj Close</th>
      <td>1009.0</td>
      <td>4.190007e+02</td>
      <td>1.082900e+02</td>
      <td>2.338800e+02</td>
      <td>3.316200e+02</td>
      <td>3.786700e+02</td>
      <td>5.090800e+02</td>
      <td>6.916900e+02</td>
    </tr>
    <tr>
      <th>Volume</th>
      <td>1009.0</td>
      <td>7.570685e+06</td>
      <td>5.465535e+06</td>
      <td>1.144000e+06</td>
      <td>4.091900e+06</td>
      <td>5.934500e+06</td>
      <td>9.322400e+06</td>
      <td>5.890430e+07</td>
    </tr>
  </tbody>
</table>
</div>



### Split the data into Train&Test set


```python
train, test = train_test_split(df, test_size = 0.2)
```


```python
test_pred = test.copy()
```


```python
train.head(10)
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
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>356</th>
      <td>2019-07-08</td>
      <td>378.190002</td>
      <td>378.250000</td>
      <td>375.359985</td>
      <td>376.160004</td>
      <td>376.160004</td>
      <td>3113400</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2018-03-01</td>
      <td>292.750000</td>
      <td>295.250000</td>
      <td>283.829987</td>
      <td>290.390015</td>
      <td>290.390015</td>
      <td>11932100</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>2022-01-28</td>
      <td>386.760010</td>
      <td>387.000000</td>
      <td>372.079987</td>
      <td>384.359985</td>
      <td>384.359985</td>
      <td>11966600</td>
    </tr>
    <tr>
      <th>384</th>
      <td>2019-08-15</td>
      <td>299.500000</td>
      <td>300.630005</td>
      <td>288.000000</td>
      <td>295.760010</td>
      <td>295.760010</td>
      <td>9629200</td>
    </tr>
    <tr>
      <th>208</th>
      <td>2018-11-30</td>
      <td>288.000000</td>
      <td>290.809998</td>
      <td>283.059998</td>
      <td>286.130005</td>
      <td>286.130005</td>
      <td>11860100</td>
    </tr>
    <tr>
      <th>484</th>
      <td>2020-01-08</td>
      <td>331.489990</td>
      <td>342.700012</td>
      <td>331.049988</td>
      <td>339.260010</td>
      <td>339.260010</td>
      <td>7104500</td>
    </tr>
    <tr>
      <th>290</th>
      <td>2019-04-02</td>
      <td>366.250000</td>
      <td>368.420013</td>
      <td>362.220001</td>
      <td>367.720001</td>
      <td>367.720001</td>
      <td>5158700</td>
    </tr>
    <tr>
      <th>412</th>
      <td>2019-09-25</td>
      <td>255.710007</td>
      <td>266.600006</td>
      <td>253.699997</td>
      <td>264.750000</td>
      <td>264.750000</td>
      <td>11643800</td>
    </tr>
    <tr>
      <th>886</th>
      <td>2021-08-12</td>
      <td>511.859985</td>
      <td>513.000000</td>
      <td>507.200012</td>
      <td>510.720001</td>
      <td>510.720001</td>
      <td>1685700</td>
    </tr>
    <tr>
      <th>343</th>
      <td>2019-06-18</td>
      <td>355.570007</td>
      <td>361.500000</td>
      <td>353.750000</td>
      <td>357.119995</td>
      <td>357.119995</td>
      <td>5428500</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head(10)
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
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>204</th>
      <td>2018-11-26</td>
      <td>260.549988</td>
      <td>266.250000</td>
      <td>253.800003</td>
      <td>261.429993</td>
      <td>261.429993</td>
      <td>12498600</td>
    </tr>
    <tr>
      <th>505</th>
      <td>2020-02-07</td>
      <td>365.040009</td>
      <td>371.799988</td>
      <td>363.570007</td>
      <td>366.769989</td>
      <td>366.769989</td>
      <td>4385200</td>
    </tr>
    <tr>
      <th>517</th>
      <td>2020-02-26</td>
      <td>366.309998</td>
      <td>382.000000</td>
      <td>365.000000</td>
      <td>379.239990</td>
      <td>379.239990</td>
      <td>8934100</td>
    </tr>
    <tr>
      <th>353</th>
      <td>2019-07-02</td>
      <td>374.890015</td>
      <td>376.000000</td>
      <td>370.309998</td>
      <td>375.429993</td>
      <td>375.429993</td>
      <td>3625000</td>
    </tr>
    <tr>
      <th>303</th>
      <td>2019-04-22</td>
      <td>359.700012</td>
      <td>377.690002</td>
      <td>359.000000</td>
      <td>377.339996</td>
      <td>377.339996</td>
      <td>11980500</td>
    </tr>
    <tr>
      <th>789</th>
      <td>2021-03-25</td>
      <td>516.989990</td>
      <td>518.530029</td>
      <td>497.000000</td>
      <td>502.859985</td>
      <td>502.859985</td>
      <td>4926800</td>
    </tr>
    <tr>
      <th>175</th>
      <td>2018-10-15</td>
      <td>337.630005</td>
      <td>339.209991</td>
      <td>326.929993</td>
      <td>333.130005</td>
      <td>333.130005</td>
      <td>11215000</td>
    </tr>
    <tr>
      <th>640</th>
      <td>2020-08-20</td>
      <td>484.690002</td>
      <td>498.940002</td>
      <td>483.890015</td>
      <td>497.899994</td>
      <td>497.899994</td>
      <td>5132500</td>
    </tr>
    <tr>
      <th>869</th>
      <td>2021-07-20</td>
      <td>526.070007</td>
      <td>536.640015</td>
      <td>520.299988</td>
      <td>531.049988</td>
      <td>531.049988</td>
      <td>6930400</td>
    </tr>
    <tr>
      <th>719</th>
      <td>2020-12-11</td>
      <td>495.000000</td>
      <td>503.339996</td>
      <td>494.850006</td>
      <td>503.220001</td>
      <td>503.220001</td>
      <td>3210900</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_train = train[['Open', 'High', 'Low', 'Volume']].values
x_test = test[['Open', 'High', 'Low', 'Volume']].values
```


```python
y_train = train['Close'].values
y_test = test['Close'].values
```

### Linear Regression


```python
model_lnr = LinearRegression()
model_lnr.fit(x_train, y_train)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
y_pred = model_lnr.predict(x_test)
```


```python
result = model_lnr.predict([[262.000000, 267.899994, 250.029999, 11896100]])
print(result)
```

    [257.5128373]
    

### Model Evaluation


```python
print("MSE",round(mean_squared_error(y_test,y_pred), 3))
print("RMSE",round(np.sqrt(mean_squared_error(y_test,y_pred)), 3))
print("MAE",round(mean_absolute_error(y_test,y_pred), 3))
print("MAPE",round(mean_absolute_percentage_error(y_test,y_pred), 3))
print("R2 Score : ", round(r2_score(y_test,y_pred), 3))
```

    MSE 13.746
    RMSE 3.708
    MAE 2.778
    MAPE 0.007
    R2 Score :  0.999
    

### Model Visualization


```python
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
```


```python
viz['Date']=pd.to_datetime(viz['Date'],format='%Y-%m-%d')
```


```python
data = pd.DataFrame(viz[['Date','Close']])
data=data.reset_index()
data=data.drop('index',axis=1)
data.set_index('Date', inplace=True)
data = data.asfreq('D')
data
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
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-02-05</th>
      <td>254.259995</td>
    </tr>
    <tr>
      <th>2018-02-06</th>
      <td>265.720001</td>
    </tr>
    <tr>
      <th>2018-02-07</th>
      <td>264.559998</td>
    </tr>
    <tr>
      <th>2018-02-08</th>
      <td>250.100006</td>
    </tr>
    <tr>
      <th>2018-02-09</th>
      <td>249.470001</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-01-31</th>
      <td>427.140015</td>
    </tr>
    <tr>
      <th>2022-02-01</th>
      <td>457.130005</td>
    </tr>
    <tr>
      <th>2022-02-02</th>
      <td>429.480011</td>
    </tr>
    <tr>
      <th>2022-02-03</th>
      <td>405.600006</td>
    </tr>
    <tr>
      <th>2022-02-04</th>
      <td>410.170013</td>
    </tr>
  </tbody>
</table>
<p>1461 rows × 1 columns</p>
</div>




```python
style()

plt.title('Closing Stock Price', color="white")
plt.plot(viz.Date, viz.Close, color="#94F008")
plt.legend(["Close"], loc ="lower right", facecolor='black', labelcolor='white')
```




    <matplotlib.legend.Legend at 0x2513a874f70>




    
![png](output_29_1.png)
    



```python
style()

plt.scatter(y_pred, y_test, color='red', marker='o')
plt.scatter(y_test, y_test, color='blue')
plt.plot(y_test, y_test, color='lime')
```




    [<matplotlib.lines.Line2D at 0x2513ea56a00>]




    
![png](output_30_1.png)
    



```python
test_pred['Close_Prediction'] = y_pred
test_pred
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
      <th>Date</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
      <th>Close_Prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>204</th>
      <td>2018-11-26</td>
      <td>260.549988</td>
      <td>266.250000</td>
      <td>253.800003</td>
      <td>261.429993</td>
      <td>261.429993</td>
      <td>12498600</td>
      <td>260.266926</td>
    </tr>
    <tr>
      <th>505</th>
      <td>2020-02-07</td>
      <td>365.040009</td>
      <td>371.799988</td>
      <td>363.570007</td>
      <td>366.769989</td>
      <td>366.769989</td>
      <td>4385200</td>
      <td>369.288521</td>
    </tr>
    <tr>
      <th>517</th>
      <td>2020-02-26</td>
      <td>366.309998</td>
      <td>382.000000</td>
      <td>365.000000</td>
      <td>379.239990</td>
      <td>379.239990</td>
      <td>8934100</td>
      <td>378.137088</td>
    </tr>
    <tr>
      <th>353</th>
      <td>2019-07-02</td>
      <td>374.890015</td>
      <td>376.000000</td>
      <td>370.309998</td>
      <td>375.429993</td>
      <td>375.429993</td>
      <td>3625000</td>
      <td>372.050813</td>
    </tr>
    <tr>
      <th>303</th>
      <td>2019-04-22</td>
      <td>359.700012</td>
      <td>377.690002</td>
      <td>359.000000</td>
      <td>377.339996</td>
      <td>377.339996</td>
      <td>11980500</td>
      <td>374.110145</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>328</th>
      <td>2019-05-28</td>
      <td>354.390015</td>
      <td>361.200012</td>
      <td>353.649994</td>
      <td>355.059998</td>
      <td>355.059998</td>
      <td>4717100</td>
      <td>359.304797</td>
    </tr>
    <tr>
      <th>361</th>
      <td>2019-07-15</td>
      <td>372.940002</td>
      <td>373.679993</td>
      <td>362.299988</td>
      <td>366.600006</td>
      <td>366.600006</td>
      <td>7944700</td>
      <td>365.184656</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018-02-13</td>
      <td>257.290009</td>
      <td>261.410004</td>
      <td>254.699997</td>
      <td>258.269989</td>
      <td>258.269989</td>
      <td>6855200</td>
      <td>258.697008</td>
    </tr>
    <tr>
      <th>808</th>
      <td>2021-04-22</td>
      <td>513.820007</td>
      <td>513.960022</td>
      <td>500.549988</td>
      <td>508.779999</td>
      <td>508.779999</td>
      <td>9061100</td>
      <td>503.584821</td>
    </tr>
    <tr>
      <th>228</th>
      <td>2019-01-02</td>
      <td>259.279999</td>
      <td>269.750000</td>
      <td>256.579987</td>
      <td>267.660004</td>
      <td>267.660004</td>
      <td>11679500</td>
      <td>266.031543</td>
    </tr>
  </tbody>
</table>
<p>202 rows × 8 columns</p>
</div>




```python
test_pred[['Close', 'Close_Prediction']].describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Close</th>
      <td>202.0</td>
      <td>418.874207</td>
      <td>108.787723</td>
      <td>233.880005</td>
      <td>328.070007</td>
      <td>379.024994</td>
      <td>507.972496</td>
      <td>682.020020</td>
    </tr>
    <tr>
      <th>Close_Prediction</th>
      <td>202.0</td>
      <td>418.441858</td>
      <td>108.519345</td>
      <td>242.480083</td>
      <td>328.099028</td>
      <td>381.373167</td>
      <td>506.669983</td>
      <td>681.987242</td>
    </tr>
  </tbody>
</table>
</div>



### Savingo the Data as CSV



```python
test_pred['Date'] = pd.to_datetime(test_pred['Date'],format='%Y-%m-%d')
```


```python
output = pd.DataFrame(test_pred[['Date', 'Close', 'Close_Prediction']])
output = output.reset_index()
output = output.drop('index',axis=1)
output.set_index('Date', inplace=True)
output =  output.asfreq('D')
output
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
      <th>Close</th>
      <th>Close_Prediction</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-02-13</th>
      <td>258.269989</td>
      <td>258.697008</td>
    </tr>
    <tr>
      <th>2018-02-14</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-02-15</th>
      <td>280.269989</td>
      <td>276.953455</td>
    </tr>
    <tr>
      <th>2018-02-16</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-02-17</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-01-31</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2022-02-01</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2022-02-02</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2022-02-03</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2022-02-04</th>
      <td>410.170013</td>
      <td>403.256070</td>
    </tr>
  </tbody>
</table>
<p>1453 rows × 2 columns</p>
</div>




```python
output.to_csv('Close_Prediction.csv', index=True)
print("CSV successfully saved!")
```

    CSV successfully saved!
    
