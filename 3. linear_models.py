# HOUSE PRICE PREDICTION LINEAR MODELS

'''
Models to be used:
    - Multiple Linear Regression
    - Lasso Regression
    - Ridge Regression
    - ElasticNet Regression
'''

# Import dependecies

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from ngboost import NGBRegressor

import os
import pickle

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


# Preparation of Dataset

# Recalling/Reloading the saved dataset
train_df = pd.read_pickle("datasets/house_prices/prepared_data/train_df.pkl")
test_df = pd.read_pickle("datasets/house_prices/prepared_data/test_df.pkl")

# Gather the dataset
all_data = [train_df, test_df]

# Define the dependent and independent variables
X = train_df.drop('SalePrice', axis=1)
y = np.ravel(train_df[["SalePrice"]])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
y_train = np.ravel(y_train)  # dimension adjustment for dependent variable


# MODELLING


# Lasso Regression

# Model and Prediction
ridge_model = Ridge().fit(X_train, y_train)
ridge_model.coef_
ridge_model.intercept_
ridge_model.alpha

# Train error
y_train_pred = ridge_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_train_pred))

# Test error
y_test_pred = ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_test_pred)) # 26178.466527526194

# Model Tuning

ridge_params = {"alpha": 10 ** np.linspace(10, -2, 100) * 0.5}
ridge_model = Ridge()
ridge_cv_model = GridSearchCV(ridge_model, ridge_params, cv=10).fit(X_train, y_train)
ridge_cv_model.best_params_ # {'alpha': 9.369087114301934}

# Final Model

ridge_tuned = Ridge(**ridge_cv_model.best_params_).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 25783.20471599487


# Lasso Regression

# Model and Prediction

lasso_model = Lasso().fit(X_train, y_train)
lasso_model.intercept_
lasso_model.coef_

# Train error
lasso_model.predict(X_train)
y_train_pred = lasso_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_train_pred))

# Train error
lasso_model.predict(X_test)
y_test_pred = lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_test_pred)) # 25579.64146404799

# Model Tuning

lasso_params = {"alpha": [1.0, 10 ** np.linspace(10, -2, 100) * 0.5]}
lasso_model = Lasso()
lasso_cv_model = GridSearchCV(lasso_model, lasso_params, cv=10).fit(X_train, y_train)
lasso_cv_model.best_params_ # {'alpha': 1.0}

# Final Model

lasso_tuned = Lasso(**lasso_cv_model.best_params_).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 25579.64146404799


# ElasticNet Regression

# Model and Prediction

enet_model = ElasticNet().fit(X_train, y_train)
enet_model.intercept_
enet_model.coef_

# Train error
enet_model.predict(X_train)
y_train_pred = enet_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_train_pred))

# Train error
enet_model.predict(X_test)
y_test_pred = enet_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_test_pred)) # 44627.4905215284

# Model Tuning

enet_params = {"l1_ratio": [0.1, 0.4, 0.5, 0.6, 0.8, 1],
               "alpha": [0.1, 0.01, 0.001, 0.2, 0.3, 0.5, 0.8, 0.9, 1]}
enet_model = ElasticNet()

enet_cv_model = GridSearchCV(enet_model, enet_params, cv=10).fit(X_train, y_train)
enet_cv_model.best_params_ # 'alpha': 0.01, 'l1_ratio': 0.1}

enet_tuned = ElasticNet(**enet_cv_model.best_params_).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 25846.591755960915


# MODELLING - RECAP


# Evaluate each model in turn by looking at train and test errors and scores
def evaluate_model(models):
    # Define lists to track names and results for models
    names = []
    train_rmse_results = []
    test_rmse_results = []
    train_r2_scores = []
    test_r2_scores = []

    print('################ RMSE and R2_score values for test set for the models: ################\n')
    for name, model in models:
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rmse_result = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse_result = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_rmse_results.append(train_rmse_result)
        test_rmse_results.append(test_rmse_result)

        train_r2_score = model.score(X_train, y_train)
        test_r2_score = model.score(X_test, y_test)
        train_r2_scores.append(train_r2_score)
        test_r2_scores.append(test_r2_score)

        names.append(name)
        msg = "%s: %f --> %f" % (name, test_rmse_result, test_r2_score)
        print(msg)

    print('\n################ Train and test results for the model: ################\n')
    data_result = pd.DataFrame({'models': names,
                                'rmse_train': train_rmse_results,
                                'rmse_test': test_rmse_results,
                                'r2_score_train': train_r2_scores,
                                'r2_score_test': test_r2_scores
                                })
    print(data_result)

    # Plot the results
    plt.figure(figsize = (15, 12))
    sns.barplot(x='rmse_test', y='models', data=data_result, color="r")
    plt.xlabel('RMSE values')
    plt.ylabel('Models')
    plt.title('RMSE For Test Set')
    plt.show()


# See the results for base models
base_models = [#('LinearRegression', LinearRegression()),
               ('Ridge', Ridge()),
               ('Lasso', Lasso()),
               ('ElasticNet', ElasticNet())]

evaluate_model(base_models)

# LinearRegression: 1794382405718603.000000 --> -495776509438438604800.000000
# Ridge: 26178.466528 --> 0.894478
# Lasso: 25579.641464 --> 0.899250
# ElasticNet: 44627.490522 --> 0.693337

# See the results for tuned models
tuned_models = [('LinearRegression', LinearRegression()),
                ('Ridge', ridge_tuned),
                ('Lasso', lasso_tuned),
                ('ElasticNet', enet_tuned)]

evaluate_model(tuned_models)

# LinearRegression: 1794382405718603.000000 --> -495776509438438604800.000000
# Ridge: 25783.204716 --> 0.897640
# Lasso: 25579.641464 --> 0.899250
# ElasticNet: 25846.591756 --> 0.897136


# Pickle Models --> Saving tuned models

# Create a folder named 'Models'
# save working directory

cur_dir = os.getcwd()
cur_dir

# change working directory:
os.chdir('projects/models')

# # Save the models
# for model in tuned_models:
#     pickle.dump(model[1], open(str(model[0]) + ".pkl", 'wb'))

# Load the model that we saved before
ridge = pickle.load(open(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\models\Ridge.pkl', 'rb'))
ridge.predict(X_test)[0:5]