```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error as mse, mean_absolute_percentage_error as mape
import warnings
warnings.filterwarnings('ignore')
```

## E2E Sample ML Project - Final Project

## Projetc Description

This project is to develop a micro-service for rent price prediction in Saint-Petersburg.

#### Data

Data used is log of rent prices in Yandex.Realty from 2014 to 2018. The initial features are


```python
initial_df.columns
```




    Index(['offer_id', 'first_day_exposition', 'last_day_exposition', 'last_price',
           'floor', 'open_plan', 'rooms', 'studio', 'area', 'kitchen_area',
           'living_area', 'agent_fee', 'renovation', 'offer_type', 'category_type',
           'unified_address', 'building_id'],
          dtype='object')



#### Used in preprocessing

The useless for model training features (eg. offer_type, building_id) are ignored in the analysis. These steps were followed in data preprocessing:

- [x] Set d-types
- [x] Manage outliers
- [x] Fill missing values
- [x] Drop inconsistent rows and remove otliers
- [x] Scale features

As the result of preprocessing there are two additional models:

1. Imputer - based on scikit-learn KNNImputer model
2. Scaler (for features only) - based on scikit-learn StandardScaler

Both of them are used in deployed version, nevertheless, imputer is used only in the first mode of price prediction, the second one returns error if it receives missing values.

#### Feature generation

There is no huge amount of features generated on-top of the initial, still, they are:

1. area_per_room - just represents average room area to let model distract multi-room tiny flat (kommunalka) from truely big one
2. living_area_ratio - the share of living area to total area
3. kitchen_area_ratio - the share of kitchen area to living area

#### Models tuning

There were 5 models assessed:

1. Constant median predictor (just to ensure adequacy of ML models trained)
2. Linear Regression
3. Decision Tree Regression
4. Random Forest Regression
5. Gradient Boosting (LGBM-driven)

3-5 models were tuned with huge grid of hyperparams using RandomizedSearchCV with negative MSE as optimization function.

#### Models deployed

Finally all the models were adequate, here is the table of resulting RMSE and MAPE:


```python
result_df.sort_values(by='RMSE', ascending=True)
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
      <th>RMSE</th>
      <th>MAPE</th>
      <th>Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>12491.365976</td>
      <td>0.203483</td>
      <td>Gradient Boosting</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12727.320239</td>
      <td>0.209839</td>
      <td>Random Forest</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12940.276583</td>
      <td>0.212151</td>
      <td>Decision Tree</td>
    </tr>
    <tr>
      <th>0</th>
      <td>13135.304648</td>
      <td>0.240026</td>
      <td>Linear Regression</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20485.751795</td>
      <td>0.447151</td>
      <td>Median Constant</td>
    </tr>
  </tbody>
</table>
</div>



So, Gradient Boosting and Random Forest were deployed as different modes of prediction.

### 1. Short EDA


```python
initial_df = pd.read_csv('real_estate.tsv', sep='\t')
```


```python
df = initial_df[initial_df['offer_type']==2]
```

###### Several columns seems unusable from start, let's drop them 


```python
unusable_cols = [
    'offer_id',
    'last_day_exposition',
    'first_day_exposition',
    'unified_address',
    'building_id',
    'offer_type',
    'category_type',
]
df.drop(columns=unusable_cols, inplace=True)
```

###### We have a lot of NaNs in 4 columns, let's get back in data prep stage, now focus on visuals and data understanding


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
      <th>last_price</th>
      <th>floor</th>
      <th>rooms</th>
      <th>area</th>
      <th>kitchen_area</th>
      <th>living_area</th>
      <th>agent_fee</th>
      <th>renovation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.711860e+05</td>
      <td>171186.000000</td>
      <td>171186.000000</td>
      <td>171186.000000</td>
      <td>126875.000000</td>
      <td>133280.000000</td>
      <td>133188.000000</td>
      <td>112603.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.171926e+04</td>
      <td>6.793727</td>
      <td>1.557890</td>
      <td>52.455625</td>
      <td>11.560852</td>
      <td>30.027255</td>
      <td>70.701652</td>
      <td>3.864542</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.249697e+04</td>
      <td>5.058804</td>
      <td>0.886736</td>
      <td>24.417140</td>
      <td>79.465616</td>
      <td>17.411222</td>
      <td>28.828281</td>
      <td>4.526333</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.300000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000e+04</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>36.000000</td>
      <td>8.000000</td>
      <td>18.000000</td>
      <td>50.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.500000e+04</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>45.000000</td>
      <td>10.000000</td>
      <td>26.000000</td>
      <td>70.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.500000e+04</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>60.000000</td>
      <td>12.000000</td>
      <td>36.000000</td>
      <td>100.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.900000e+06</td>
      <td>92.000000</td>
      <td>5.000000</td>
      <td>200.000000</td>
      <td>25000.000000</td>
      <td>2015.000000</td>
      <td>100.000000</td>
      <td>11.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure()
plt.boxplot(df.last_price)
plt.show()
```


    
![png](output_14_0.png)
    



```python
df = df[df.last_price<=3*10**5]
```


```python
plt.figure()
plt.boxplot(df.last_price)
plt.show()
```


    
![png](output_16_0.png)
    



```python
plt.figure()
plt.title('last_price')
df.last_price.hist(bins=100)
plt.show()

plt.figure()
plt.title('floor')
df.floor.hist(bins=100)
plt.show()

plt.figure()
plt.title('rooms')
df.rooms.hist(bins=30)
plt.show()

plt.figure()
plt.title('area')
df.area.hist(bins=30)
plt.show()

plt.figure()
plt.title('kitchen_area')
df.kitchen_area.hist(bins=30)
plt.show()


plt.figure()
plt.title('living_area')
df.living_area.hist(bins=30)
plt.show()
```


    
![png](output_17_0.png)
    



    
![png](output_17_1.png)
    



    
![png](output_17_2.png)
    



    
![png](output_17_3.png)
    



    
![png](output_17_4.png)
    



    
![png](output_17_5.png)
    


###### So, we have a lot of outliers in several fields, let's manage them within data prep stage. But before it, let's check one trick.


```python
df[(df['area']<df['living_area'])|(df['area']<df['kitchen_area'])].count()
```




    last_price      20
    floor           20
    open_plan       20
    rooms           20
    studio          20
    area            20
    kitchen_area    18
    living_area     18
    agent_fee       11
    renovation      15
    dtype: int64



###### Yep, thick is present. 37 rows have total area less than kitchen_area or living_area. This is inconcistent and to be managed.

### 2. Data preparation

#### Short plan:

- [x] Set d-types
- [x] Fill missing values
- [x] Drop inconsistent rows and remove otliers
- [x] Scale features



```python
df['area_per_room'] = df['area'] / (df['rooms'].replace(0,1))
df['living_area_ratio'] = df['living_area'] / df['area']
df['kitchen_area_ratio'] = df['kitchen_area'] / df['living_area']
```


```python
df = df[ #dropping rows with inconcistencies about area, excluding rows with missing values
    (
        (df['area']>=df['living_area'])|
        (df['living_area'].isna())
        
    )&
    (
        (df['area']>=df['kitchen_area'])|
        (df['kitchen_area'].isna())
    )
]
```


```python
df = df[df['area'] <=150]
```


```python
df.renovation.fillna(0, inplace=True) #filling NaNs in rennovation and agent_fee with zero, 
df.agent_fee.fillna(0, inplace=True) #seems to be logical that missing value means zero
```


```python
for col in ['open_plan', 'studio', 'renovation']: #setting integer d-type for binary fields
    df[col] = df[col].astype('int')
```


```python
train, test = train_test_split(df, random_state=42, test_size=0.2) #splitting dataset into train and test
train_x = train.drop(columns=['last_price']).reset_index(drop=True) #distinguishing target from features
train_y = train['last_price'].reset_index(drop=True)
test_x = test.drop(columns=['last_price']).reset_index(drop=True)
test_y = test['last_price'].reset_index(drop=True)
```


```python
scaler = StandardScaler() #scaling features for better quality of prediction
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
```


```python
imputer = KNNImputer(n_neighbors=2) #imputing missing values using k-nearest-neighbors approach
train_x = imputer.fit_transform(train_x)
test_x = imputer.transform(test_x)
```

### 3. Model training


```python
# models to be assessed
tree = DecisionTreeRegressor(random_state = 42)
forest = RandomForestRegressor(random_state = 42)
lreg = LinearRegression(normalize = False)
lgbm = LGBMRegressor(random_state = 42, n_jobs = -1, silent = True)

#hyperparsms grids to be considered
tree_grid = {'max_depth': range(1, 10)}
forest_grid = {'max_depth': range(1, 10), 'n_estimators': range(1, 100, 10)}
lgbm_grid = {'n_estimators': range(1, 200, 5), 'max_depth': range(1, 100), 
            'learning_rate': [0,0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5], 'reg_alpha': [0,0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5], 
            'num_leaves': range(0, 50, 10), 'reg_lambda': [0,0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5]}

#objects for tuning
best_tree = RandomizedSearchCV(tree, n_jobs = -1, n_iter = 9, param_distributions = tree_grid, random_state = 42, scoring = 'neg_mean_squared_error', cv = 5)
best_forest = RandomizedSearchCV(forest, n_jobs = -1, n_iter = 9, param_distributions = forest_grid, random_state = 42, scoring = 'neg_mean_squared_error', cv = 5)
best_sgb = RandomizedSearchCV(lgbm, n_jobs = -1, n_iter = 9, param_distributions = lgbm_grid, random_state = 42, scoring = 'neg_mean_squared_error', cv = 5)
```


```python
lreg.fit(train_x, train_y)
rmse_lreg = mse(lreg.predict(test_x), test_y)**(1/2)
mape_lreg = mape(lreg.predict(test_x), test_y)
print(f'RMSE of linear regression {int(rmse_lreg)}')
print(f'MAPE of linear regression {round(mape_lreg*100, 3)}%')
```

    RMSE of linear regression 13135
    MAPE of linear regression 24.003%



```python
best_tree.fit(train_x, train_y)
rmse_tree = mse(best_tree.predict(test_x), test_y)**(1/2)
mape_tree = mape(best_tree.predict(test_x), test_y)
print(f'RMSE of decision tree {int(rmse_tree)}')
print(f'MAPE of decision tree {round(mape_tree*100, 3)}%')
```

    RMSE of decision tree 12940
    MAPE of decision tree 21.215%



```python
best_forest.fit(train_x, train_y)
rmse_forest = mse(best_forest.predict(test_x), test_y)**(1/2)
mape_forest = mape(best_forest.predict(test_x), test_y)
print(f'RMSE of random forest {int(rmse_forest)}')
print(f'MAPE of random forest {round(mape_forest*100, 3)}%')
```

    RMSE of random forest 12727
    MAPE of random forest 20.984%



```python
best_sgb.fit(train_x, train_y)
rmse_sgb = mse(best_sgb.predict(test_x), test_y)**(1/2)
mape_sgb = mape(best_sgb.predict(test_x), test_y)
print(f'RMSE of gradient boosting {int(rmse_sgb)}')
print(f'MAPE of gradient boosting {round(mape_sgb*100, 3)}%')
```

    RMSE of gradient boosting 12491
    MAPE of gradient boosting 20.348%



```python
median = train_y.median()
rmse_median = mse(np.array([median]*len(test_x)), test_y)**(1/2)
mape_median = mape(np.array([median]*len(test_x)), test_y)
print(f'RMSE of median constant prediction {int(rmse_median)}')
print(f'MAPE of median constant prediction {round(mape_median*100, 3)}%')
```

    RMSE of median constant prediction 20485
    MAPE of median constant prediction 44.715%



```python
rmse_scores = [
    rmse_lreg,
    rmse_tree,
    rmse_forest,
    rmse_sgb,
    rmse_median
]

mape_scores = [
    mape_lreg,
    mape_tree,
    mape_forest,
    mape_sgb,
    mape_median
]

models = [
    'Linear Regression',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting',
    'Median Constant'
]

result_df = pd.DataFrame({'RMSE':rmse_scores, 'MAPE':mape_scores, 'Model':models})
```


```python
result_df.sort_values(by='RMSE', ascending=True)
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
      <th>RMSE</th>
      <th>MAPE</th>
      <th>Model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>12491.365976</td>
      <td>0.203483</td>
      <td>Gradient Boosting</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12727.320239</td>
      <td>0.209839</td>
      <td>Random Forest</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12940.276583</td>
      <td>0.212151</td>
      <td>Decision Tree</td>
    </tr>
    <tr>
      <th>0</th>
      <td>13135.304648</td>
      <td>0.240026</td>
      <td>Linear Regression</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20485.751795</td>
      <td>0.447151</td>
      <td>Median Constant</td>
    </tr>
  </tbody>
</table>
</div>



So, here we see that all the models are adequate in comparison with median constant prediction. Still, random forest model demonstrates best score.

### Result

The best model is random forest. What else to do is to re-train the models used with whole dataset and extract them for use in production.


```python
df_x = df.drop(columns=['last_price'])
df_y = df['last_price']
```


```python
scaler = StandardScaler()
df_x = scaler.fit_transform(df_x)
imputer = SimpleImputer(strategy='mean')
df_x = imputer.fit_transform(df_x)
final_sgb = RandomizedSearchCV(lgbm, n_jobs = -1, n_iter = 9, param_distributions = lgbm_grid, random_state = 42, scoring = 'neg_mean_squared_error', cv = 5)
final_sgb.fit(df_x, df_y)
final_forest = RandomizedSearchCV(forest, n_jobs = -1, n_iter = 9, param_distributions = forest_grid, random_state = 42, scoring = 'neg_mean_squared_error', cv = 5)
final_forest.fit(df_x, df_y)
```




    RandomizedSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
                       n_iter=9, n_jobs=-1,
                       param_distributions={'max_depth': range(1, 10),
                                            'n_estimators': range(1, 100, 10)},
                       random_state=42, scoring='neg_mean_squared_error')




```python
import joblib
sgb_file = 'model.pkl'
forest_file = 'model2.pkl'
scaler_file = 'scaler.pkl'
imputer_file = 'imputer.pkl'
joblib.dump(scaler, scaler_file)
joblib.dump(imputer, imputer_file)
joblib.dump(final_sgb, sgb_file)
joblib.dump(final_forest, forest_file)
```




    ['model2.pkl']




```python

```
