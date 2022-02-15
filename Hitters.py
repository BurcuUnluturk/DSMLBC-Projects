#######################################
# Hitters
#######################################

!pip install xgboost
!pip install lightgbm
conda install -c conda-forge lightgbm
!pip install catboost

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate

from helpers.data_prep import *
from helpers.eda import *


df = pd.read_csv("C:/Users/burcu3153/PycharmProjects/dsmlbc5/datasets/hitters.csv")
df.head()


#######################################
# Quick Data Preprocessing
#######################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# cat_cols
# ['League', 'Division', 'NewLeague']

# num_cols
# ['AtBat', 'Hits', 'HmRun', 'Runs', 'RBI', 'Walks', 'Years', 'CAtBat', 'CHits', 'CHmRun', 'CRuns',
# 'CRBI', 'CWalks', 'PutOuts', 'Assists', 'Errors', 'Salary']

df.describe().T

"""
        count    mean     std   min    25%     50%     75%      max
AtBat   322.00  380.93  153.40 16.00 255.25  379.50  512.00   687.00
Hits    322.00  101.02   46.45  1.00  64.00   96.00  137.00   238.00
HmRun   322.00   10.77    8.71  0.00   4.00    8.00   16.00    40.00
Runs    322.00   50.91   26.02  0.00  30.25   48.00   69.00   130.00
RBI     322.00   48.03   26.17  0.00  28.00   44.00   64.75   121.00
Walks   322.00   38.74   21.64  0.00  22.00   35.00   53.00   105.00
Years   322.00    7.44    4.93  1.00   4.00    6.00   11.00    24.00
CAtBat  322.00 2648.68 2324.21 19.00 816.75 1928.00 3924.25 14053.00
CHits   322.00  717.57  654.47  4.00 209.00  508.00 1059.25  4256.00
CHmRun  322.00   69.49   86.27  0.00  14.00   37.50   90.00   548.00
CRuns   322.00  358.80  334.11  1.00 100.25  247.00  526.25  2165.00
CRBI    322.00  330.12  333.22  0.00  88.75  220.50  426.25  1659.00
CWalks  322.00  260.24  267.06  0.00  67.25  170.50  339.25  1566.00
PutOuts 322.00  288.94  280.70  0.00 109.25  212.00  325.00  1378.00
Assists 322.00  106.91  136.85  0.00   7.00   39.50  166.00   492.00
Errors  322.00    8.04    6.37  0.00   3.00    6.00   11.00    32.00
Salary  263.00  535.93  451.12 67.50 190.00  425.00  750.00  2460.00
"""

# null values
df.isnull().sum()
# Salary  59

# NA  with mean value
df['Salary'].fillna(df['Salary'].median(), inplace=True)
df.isnull().sum()

#AAre there any outliers? If yes set values with upper and lower thresholds
grab_outliers(df,'Salary')
check_outlier(df, 'Salary')
# True
outlier_thresholds(df, 'Salary')
#(-650.0, 1590.0)
replace_with_thresholds(df,'Salary')
check_outlier(df, 'Salary')
# False

for col in cat_cols:
    df.loc[:, col] = label_encoder(df, col)

corr = df.corr()
sns.heatmap(corr,
         xticklabels=corr.columns,
         yticklabels=corr.columns)
plt.show()

# Hits, Runs, RBI, Walks, AtBat
# Years, CatBat, Chits, CHmRuns, CRBI, CWalks
# Salary, CatBat, CHits, CmRuns, CRBI

# Hits/Chits (This years successful hits/All the time successful hits)
# Chits/CatBat (Sucessful hits)

df['CHits/CAtBat'] = df['CHits'] / df['CAtBat']
df['Hits/CHits'] = df['Hits'] / df['CHits']
df['CAtBat/Years'] = df['CAtBat'] / df['Years']
df['CHits/Years'] = df['CHits'] / df['Years']
df['CHmRun/Years'] = df['CHmRun'] / df['Years']
df['CRBI/Years'] = df['CRBI'] / df['Years']

# Labeling
# Label encoder would have worked here as the cat has 2 unique values
# ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
# df = one_hot_encoder(df, ohe_cols)


check_df(df)

y = df["Salary"]
X = df.drop(["Salary"], axis=1)


######################################################
# Base Models
######################################################


models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


"""
RMSE: 247.3362 (LR) 
RMSE: 247.2512 (Ridge) 
RMSE: 245.8074 (Lasso) 
RMSE: 267.2508 (ElasticNet) 
RMSE: 268.4303 (KNN) 
RMSE: 316.6195 (CART) 
RMSE: 229.2325 (RF) 
RMSE: 338.7543 (SVR) 
RMSE: 229.1104 (GBM) 
RMSE: 252.682 (XGBoost)
RMSE: 228.3235 (LightGBM)  
"""


y.mean()
#496
y.std()
#345


######################################################
# Automated Hyperparameter Optimization
######################################################


cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [3,5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.5, 0.7, 1]}

regressors = [("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]



best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

"""
########## CART ##########
RMSE: 325.8875 (CART) 
RMSE (After): 256.1081 (CART) 
CART best params: {'max_depth': 3, 'min_samples_split': 2}
########## RF ##########
RMSE: 228.7444 (RF) 
RMSE (After): 221.9778 (RF) 
RF best params: {'max_depth': 5, 'max_features': 5, 'min_samples_split': 8, 'n_estimators': 200}
########## XGBoost ##########
RMSE: 252.682 (XGBoost) 
RMSE (After): 227.6952 (XGBoost) 
XGBoost best params: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
########## LightGBM ##########
RMSE: 228.3235 (LightGBM) 
RMSE (After): 222.1316 (LightGBM) 
LightGBM best params: {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'n_estimators': 500}
"""

######################################################
# # Stacking & Ensemble Learning
######################################################

voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"])])

voting_reg.fit(X, y)

"""
VotingRegressor(estimators=[('RF',
                             RandomForestRegressor(max_depth=5, max_features=5,
                                                   min_samples_split=8,
                                                   n_estimators=200)),
                            ('LightGBM',
                             LGBMRegressor(colsample_bytree=0.7,
                                           learning_rate=0.01,
                                           n_estimators=500))])
"""


np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))
# 218.70

