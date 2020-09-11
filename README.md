# house_price_prediction
The Ames Housing dataset is taken from kaggle competition.
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
The aim of the project is to predict house price for houses in Boston Housing Dataset.
Two files, train and test are provided and the price of the test data is to be estimated.
Here I have used XGBoost for prediction.
Thanks to Krish Naik for making these amazing videos for understanding and implementing House Price Prediction.
https://www.youtube.com/watch?v=vtm35gVP8JU
https://www.youtube.com/watch?v=htXcQPsaGWw&list=PLZoTAELRMXVPiKOxbwaniXjHJ02bdkLWy&index=2

 I will be adding exploratory data analysis and compare the XGBoost model's result with other regression techniques later.
 
## Steps in House Price Prediction  

1. Loading data  
2. Data Exploration  
    2.1 Features with Null value. 
    2.2 Numerical Features
        2.2.1 Year Features  
        2.2.3 Discrete Features  
        2.2.4 Continous Features  
    2.3 Categorical Features
3. Data Cleaning
4. Data transformation
5. Base Model Performance (XGBoost)
6. Hyperparameter Tuning
7. Final Model
8. Visualize Results

### 1. Loading data

```ruby
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
```
### 2. Data Exploration
```ruby
df.head()
df_test.shape
df.shape
# Checking the null values
df.isnull().sum()
df_test.isnull().sum()
# heatmap for visualizing the null vaues
sns.heatmap(df.isnull(),yticklabels = False,cbar = False)
df.info()

```
### 3. Data Cleaning (Handling missing values)
Filling the missing value we need to see the test and train data simultaneously.  
We will be replacing the null values with mode for categorical values and mean for continous values.  
We will delete columns with more than 50% null values as the available information add no value for our model.  

```ruby
# Handling missing data for MSZoning
df['MSZoning'].value_counts()
df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0]) # replacing with mode
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean()) # replacing with mean for train
df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean()) # replacing with mean for train
```
Please refer the notebook for other features' missing value handling

#### Dropping the columns which have lot of missing values
```ruby
df.drop(['PoolQC'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for train
df_test.drop(['PoolQC'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for test
df.drop(['Fence'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for train
df_test.drop(['Fence'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for test
df.drop(['MiscFeature'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for train
df_test.drop(['MiscFeature'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for test
df.drop(['FireplaceQu'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for train
df_test.drop(['FireplaceQu'],axis = 1, inplace = True) # dropping the Alley features as it has a lot of missing values for test

```
### 4. Data Transformation (Converting categorical variables to one hot encoding)

```ruby
# function to convert categorical variables to one hot encoding
def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final

```

```ruby
# making a copy of dataframe for future use
main_df=df.copy()
# concanating the test and train files to implement one hot encoding
final_df=pd.concat([df,df_test],axis=0)
final_df=category_onehot_multcols(columns)
```


```ruby
# removing duplicated columns
final_df =final_df.loc[:,~final_df.columns.duplicated()]
```
```ruby
# separating the test and training data
df_Train=final_df.iloc[:1459,:]
df_Test=final_df.iloc[1459:,:]
```
```ruby
# dropping the "SalePrice" column from test data
df_Test.drop(['SalePrice'],axis=1,inplace=True)
```
```ruby
# preparing data for feeding into model
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
```
### 5. Base Model Performance (XGBoost)
```ruby
# implementing XGBoost regressor
import xgboost
classifier=xgboost.XGBRegressor()
classifier.fit(X_train,y_train)
# predicting the house prices
y_predict = classifier.predict(df_Test)
```
```ruby
##Creating Sample Submission file
pred=pd.DataFrame(y_predict)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sub1.csv',index=False)
```
#### Got an RMSE score of 0.14065 on kaggle leaderboard
### 6. Hyperparameter Tuning

```ruby

regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]

## Hyper Parameter Optimization

n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]

learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Defining the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }

```
```ruby
# Set up the random search with 4-fold cross validation
from sklearn.model_selection import RandomizedSearchCV

regressor = xgboost.XGBRegressor()
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
random_cv.fit(X_train,y_train)
# finding the best estimate
random_cv.best_estimator_

```
### 7. Final Model
```ruby
# substituting the best parameters
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=2,
             min_child_weight=1, missing=None, monotone_constraints='()',
             n_estimators=900, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
regressor.fit(X_train,y_train)
y_predict1 = regressor.predict(df_Test)
```

```ruby
##Create Sample Submission file and Submit using ANN
pred=pd.DataFrame(y_predict1)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('housepricetuned.csv',index=False)
```
#### Got an RMSE score of 0.14036 on kaggle leaderboard
### Results

#### We got a slight improvement with hyperparameter tuning (from 0.14065 to 0.14036). Needs more tuning!
