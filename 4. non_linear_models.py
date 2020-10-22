# HOUSE PRICE PREDICTION LINEAR MODELS

'''
Models to be used:
    - K-Nearest Neighbors Regression
    - Support Vector Machines
    - Artificial Neural Network Models
    - Classification and Regression Trees - DecisionTreeRegressor
    - RandomForestRegressor
    - BaggingRegressor
    - Gradient Boosting Regressor
    - AdaBoostRegressor
    - XGBoost - XGBRegressor
    - LightGBM - LGBMRegressor
    - CatBoost - CatBoostRegressor
    - NGBoost - NGBRegressor
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
train_df = pd.read_pickle(r"C:\Users\yakup\PycharmProjects\dsmlbc\datasets\house_prices\prepared_data\train_df_.pkl")
test_df = pd.read_pickle(r"C:\Users\yakup\PycharmProjects\dsmlbc\datasets\house_prices\prepared_data\test_df_.pkl")

# Gather the dataset
all_data = [train_df, test_df]

# Define the dependent and independent variables
X = train_df.drop('SalePrice', axis=1)
y = np.ravel(train_df[["SalePrice"]])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
y_train = np.ravel(y_train)  # dimension adjustment for dependent variable


# Define a function to plot feature_importances
def plot_feature_importances(tuned_model):
    Importance = pd.DataFrame({'Importance': tuned_model.feature_importances_ * 100, 'Feature': X_train.columns})
    plt.figure(figsize=(10, 30))
    sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance", ascending=False))
    plt.title('Feature Importance - ') # TODO tuned_model.__name__
    plt.show()


# MODELLING


# K NEAREST NEIGHBORS

# Model and Prediction

knn_model = KNeighborsRegressor().fit(X_train, y_train)
knn_model

y_pred = knn_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 33440.72774515555

# Model Tuning

knn_params = {"n_neighbors": np.arange(2, 30, 1)}
knn_model = KNeighborsRegressor()

knn_cv_model = GridSearchCV(knn_model, knn_params, cv=10).fit(X_train, y_train)
knn_cv_model.best_params_ # 3

# Final Model
knn_tuned = KNeighborsRegressor(**knn_cv_model.best_params_).fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 32782.91159919253


# SUPPORT VECTOR MACHINES

# Model and Prediction

svr_model = SVR("linear").fit(X_train, y_train)
svr_model

y_pred = svr_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 81782.74290764822

# SVR Tuning

svr_model = SVR("linear")
svr_params = {"C": [0.01, 0.001, 0.2, 0.1, 0.5, 0.8, 0.9, 1, 10, 100, 500, 1000]}

svr_cv_model = GridSearchCV(svr_model, svr_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
svr_cv_model.best_params_ # 1000

# Final Model

svr_tuned = SVR("linear", C=1000).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 30170.424713359764


# NON-Linear SVR

# Model and Prediction

svr_model = SVR()
svr_params = {"C": [0.01, 0.001, 0.2, 0.1, 0.5, 0.8, 0.9, 1, 10, 100, 500, 1000]}
svr_cv_model = GridSearchCV(svr_model, svr_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
svr_cv_model.best_params_ # 1000

# Final Model
svr_tuned = SVR(**svr_cv_model.best_params_).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 64973.75961849786


# ARTIFICIAL NEURAL NETWORKS

# Model and Prediction

mlp_model = MLPRegressor().fit(X_train, y_train)
y_pred = mlp_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 191495.958151712

# Model Tuning

mlp_params = {"alpha": [0.1, 0.01, 0.02, 0.001, 0.0001],
              "hidden_layer_sizes": [(10, 20), (5, 5), (100, 100), (1000, 100, 10)],
              "solver": ['lbfgs', 'sgd', 'adam'],
              #"alpha": [10 ** np.linspace(10, -2, 100) * 0.5]
              }
mlp_model = MLPRegressor().fit(X_train, y_train)

mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv=10, verbose=2, n_jobs=-1).fit(X_train, y_train)
mlp_cv_model.best_params_ = {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 10)}

# Final Model
mlp_tuned = MLPRegressor(**mlp_cv_model.best_params_).fit(X_train, y_train)
y_pred = mlp_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 41830.07260225841


# CART

# Model and Prediction

cart_model = DecisionTreeRegressor(random_state=52)
cart_model.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 43898.11182793879

# Model Tuning

cart_params = {"max_depth": [2, 3, 4, 5, 10, 20, 100, 1000],
              "min_samples_split": [2, 10, 5, 30, 50, 10],
              "criterion" : ["mse", "friedman_mse", "mae"]}

cart_model = DecisionTreeRegressor()
cart_cv_model = GridSearchCV(cart_model, cart_params, cv=10).fit(X_train, y_train)
cart_cv_model.best_params_  # {'criterion': 'friedman_mse', 'max_depth': 5, 'min_samples_split': 5}


# Final Model

cart_tuned = DecisionTreeRegressor(**cart_cv_model.best_params_).fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 42498.96764584228

# Decision Rules

from skompiler import skompile
print(skompile(cart_tuned.predict).to('python/code'))

# Feature Importances

Importance = pd.DataFrame({'Importance':cart_tuned.feature_importances_*100}, index = X_train.columns)
Importance.sort_values(by = 'Importance', axis = 0, ascending = True).plot(kind = 'barh', color = 'r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
plt.show()


# RANDOM FORESTS

# Model and Prediction

rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 27078.812027499054

# Model Tuning

rf_params = {"max_depth": [3, 5, 8, 10, 15, None],
           "max_features": [5, 10, 15, 20, 50, 100],
           "n_estimators": [200, 500, 1000],
           "min_samples_split": [2, 5, 10, 20, 30, 50]}

rf_cv_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_ # {'max_depth': 15, 'max_features': 100, 'min_samples_split': 2, 'n_estimators': 1000}

# Final Model

rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 24504.87147913051

# Feature Importance

plot_feature_importances(rf_tuned)


# Bagging Regressor

# Model and Prediction

bag_model = BaggingRegressor(random_state=42).fit(X_train, y_train)
y_pred = bag_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 30524.62197148652

# Model Tuning

bag_params = {"n_estimators": range(2,20)}

bag_cv_model = GridSearchCV(bag_model, bag_params, cv=10, n_jobs=-1, verbose=1).fit(X_train, y_train)
bag_cv_model.best_params_ # {'n_estimators': 19}

# Final Model

bag_tuned = BaggingRegressor(**bag_cv_model.best_params_).fit(X_train, y_train)
y_pred = bag_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 27920.3846610151

# Feature Importance

plot_feature_importances(bag_tuned)


# GradientBoostingRegressor

# Model and Prediction

gbm_model = GradientBoostingRegressor().fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 24580.313014402054

# Model Tuning

gbm_params = {"learning_rate": [0.001, 0.1, 0.01, 0.05],
              "max_depth": [3, 5, 8, 10, 20, 30],
              "n_estimators": [200, 500, 1000, 1500, 5000],
              "subsample": [1, 0.4, 0.5, 0.7],
              "loss": ["ls", "lad", "quantile"]}

gbm_model = GradientBoostingRegressor()
gbm_cv_model = GridSearchCV(gbm_model, gbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
gbm_cv_model.best_params_ # {'learning_rate': 0.01, 'max_depth': 8, 'n_estimators': 1000, 'subsample': 0.5}

# Final Model

gbm_tuned = GradientBoostingRegressor(**gbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = gbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 21693.081005924025

# Feature Importance

plot_feature_importances(gbm_tuned)


# AdaBoostRegressor

ada_model = AdaBoostRegressor().fit(X_train, y_train)
y_pred = ada_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 35611.36852632631

# Model Tuning

ada_model = AdaBoostRegressor()
ada_params = {"learning_rate": [0.01, 0.1, 0.5, 1],
              "loss": ["linear", "square", "exponential"],
              "n_estimators": [20, 40, 100, 500, 1000]}

ada_cv_model = GridSearchCV(ada_model, ada_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
ada_cv_model.best_params_ # {'learning_rate': 0.5, 'loss': 'linear', 'n_estimators': 40}

# Final Model

ada_tuned = AdaBoostRegressor(**ada_cv_model.best_params_).fit(X_train, y_train)
y_pred = ada_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 35261.25123576798

# Feature Importance

plot_feature_importances(ada_tuned)

# from yellowbrick.model_selection import feature_importances
# feature_importances(AdaBoostRegressor(), X_test, y_test)


# XGBoost

# Model and Prediction

xgb_model = XGBRegressor().fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 26777.163722217178

# Model Tuning

xgb_params = {"learning_rate": [0.1, 0.01, 0.5],
             "max_depth": [5, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000],
             "colsample_bytree": [0.4, 0.7, 1]}

xgb_cv_model = GridSearchCV(xgb_model, xgb_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
xgb_cv_model.best_params_ # {'colsample_bytree': 0.4, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 1000}

# Final Model
xgb_tuned = XGBRegressor(**xgb_cv_model.best_params_).fit(X_train, y_train)
y_pred = xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 22613.514160521838

# Feature Importance

plot_feature_importances(xgb_tuned)


# LightGBM

# Model and Prediction

lgbm_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 24466.61545191471

# Model Tuning

lgbm_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.001, 0.1, 0.5, 1],
              "n_estimators": [200, 500, 1000, 5000],
              "max_depth": [6, 8, 10, 15, 20],
              "colsample_bytree": [1, 0.8, 0.5, 0.4]}


lgbm_cv_model = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=2).fit(X_train, y_train)
lgbm_cv_model.best_params_ # {'colsample_bytree': 0.4, 'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 5000}

# Final Model

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 21418.842759852396

# Feature Importances

plot_feature_importances(lgbm_tuned)


# CatBoost

# Model and Prediction

catb_model = CatBoostRegressor(verbose=False).fit(X_train, y_train)
y_pred = catb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 21122.3517877923

# Model Tuning

catb_model = CatBoostRegressor()
catb_params = {"iterations": ['None', 200, 500],
               "learning_rate": ['None', 0.01, 0.1],
               "depth": ['None', 3, 6]}

catb_cv_model = GridSearchCV(catb_model, catb_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
catb_cv_model.best_params_ # {'depth': 6, 'iterations': 500, 'learning_rate': 0.1}

# Final Model

catb_tuned = CatBoostRegressor(**catb_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, y_pred)) # 21122.3517877923

# Feature Importances

plot_feature_importances(catb_tuned)


# NGBoost

# Model and Prediction

ngb_model = NGBRegressor().fit(X_train, y_train)
y_pred = ngb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # 26083.30543737836

# Model Tuning

b1 = DecisionTreeRegressor(criterion="friedman_mse", max_depth=2)
b2 = DecisionTreeRegressor(criterion="friedman_mse", max_depth=4)
b3 = Ridge(alpha=0.0)

ngb_params = {"n_estimators": [20, 50, 100],
            "minibatch_frac": [1.0, 0.5, 0.2],
            "Base": [b1, b2, b3]}


ngb_model = NGBRegressor()
ngb_cv_model = GridSearchCV(ngb_model, ngb_params, cv=5, n_jobs=-1, verbose=2).fit(X_train, y_train)
ngb_cv_model.best_params_ # {'Base': Ridge(alpha=0.0), 'minibatch_frac': 0.2, 'n_estimators': 100}

# dir(ngb_cv_model)

# Final Model

ngb_tuned = NGBRegressor(**ngb_cv_model.best_params_).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, y_pred)) # 26083.30543737836


# MODELING - RECAP

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
    plt.figure(figsize=(15, 12))
    sns.barplot(x='rmse_test', y='models', data=data_result.sort_values(by="rmse_test", ascending=False), color="r")
    plt.xlabel('RMSE values')
    plt.ylabel('Models')
    plt.title('RMSE For Test Set')
    plt.show()


# See the results for base models
base_models = [#('LinearRegression', LinearRegression()),
               ('Ridge', Ridge()),
               ('Lasso', Lasso()),
               ('ElasticNet', ElasticNet()),
               ('KNN', KNeighborsRegressor()),
               ('SVR', SVR()),
               ('ANN', MLPRegressor()),
               ('CART', DecisionTreeRegressor()),
               ('BaggedTrees', BaggingRegressor()),
               ('RF', RandomForestRegressor()),
               ('AdaBoost', AdaBoostRegressor()),
               ('GBM', GradientBoostingRegressor()),
               ("XGBoost", XGBRegressor()),
               ("LightGBM", LGBMRegressor()),
               ("CatBoost", CatBoostRegressor(verbose=False)),
               ("NGBoost", NGBRegressor(verbose=False))]

evaluate_model(base_models)

# ################ RMSE and R2_score values for test set for the models: ################
# Ridge: 26587.106537 --> 0.891158
# Lasso: 25774.323220 --> 0.897711
# ElasticNet: 44016.810102 --> 0.701673
# KNN: 33440.727745 --> 0.827810
# SVR: 83258.632965 --> -0.067369
# ANN: 192407.980530 --> -4.700358
# CART: 42795.146068 --> 0.718003
# RF: 26964.928572 --> 0.888042
# BaggedTrees: 30017.952824 --> 0.861255
# GBM: 25052.391110 --> 0.903361
# AdaBoost: 35150.925735 --> 0.809748
# XGBoost: 26777.163722 --> 0.889596
# LightGBM: 24466.615452 --> 0.907827
# CatBoost: 21122.351788 --> 0.931303
# NGBoost: 25978.981860 --> 11.432905
# ################ Train and test results for the model: ################
#          models  rmse_train  rmse_test  r2_score_train  r2_score_test
# 0         Ridge   21952.880  26587.107           0.923          0.891
# 1         Lasso   19249.418  25774.323           0.941          0.898
# 2    ElasticNet   44581.388  44016.810           0.682          0.702
# 3           KNN   30483.942  33440.728           0.852          0.828
# 4           SVR   81108.965  83258.633          -0.051         -0.067
# 5           ANN  189056.268 192407.981          -4.711         -4.700
# 6          CART     155.176  42795.146           1.000          0.718
# 7            RF   12740.693  26964.929           0.974          0.888
# 8   BaggedTrees   16574.607  30017.953           0.956          0.861
# 9           GBM   14695.885  25052.391           0.965          0.903
# 10     AdaBoost   28685.120  35150.926           0.869          0.810
# 11      XGBoost    1294.504  26777.164           1.000          0.890
# 12     LightGBM   11983.002  24466.615           0.977          0.908
# 13     CatBoost    6094.709  21122.352           0.994          0.931
# 14      NGBoost   16682.243  25978.982          10.920         11.433

# See the results for tuned models
tuned_models = [('KNN', knn_tuned),
                ('SVR', svr_tuned),
                ('ANN', mlp_tuned),
                ('CART', cart_tuned),
                ('BaggedTrees', bag_tuned),
                ('RF', rf_tuned),
                ('AdaBoost', ada_tuned),
                ('GBM', gbm_tuned),
                ("XGBoost", xgb_tuned),
                ("LightGBM", lgbm_tuned),
                ("CatBoost", catb_tuned),
                ("NGBoost", ngb_tuned)]

evaluate_model(tuned_models)

# ################ Train and test results for the model: ################
#          models  rmse_train  rmse_test  r2_score_train  r2_score_test
# 0           KNN   25836.888  32782.912           0.893          0.835
# 1           SVR   63784.293  64973.760           0.350          0.350
# 2           ANN   37258.933  36952.863           0.778          0.790
# 3          CART   29090.265  42498.968           0.865          0.722
# 4            RF   11516.015  24736.952           0.979          0.906
# 5   BaggedTrees   14385.176  28347.362           0.967          0.876
# 6           GBM    3415.008  21248.074           0.998          0.930
# 7      AdaBoost   27961.202  34549.800           0.875          0.816
# 8       XGBoost    8239.261  22613.514           0.989          0.921
# 9      LightGBM    4245.021  21418.843           0.997          0.929
# 10     CatBoost    4750.250  21131.962           0.996          0.931
# 11      NGBoost   27888.618  31252.979          11.344         11.544


# Pickle Models --> Saving tuned models

# Create a folder named 'Models'
# save working directory

cur_dir = os.getcwd()
cur_dir

# change working directory:
os.chdir('projects/models')

# # Save the models
# for model in tuned_models:
#     pickle.dump(model[1], open(str(model[0]) + "_2.pkl", 'wb'))

# Load the model that we saved before
gbm = pickle.load(open(r'C:\Users\yakup\PycharmProjects\dsmlbc\projects\models\GBM.pkl', 'rb'))
gbm.predict(X_test)[0:5]


