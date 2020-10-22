# HOUSE PRICE PREDICTION MODEL

# DATA PREPROCESSING AND FEATURE ENGINEERING

# Import dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Load the dataset and gather train and test sets
train = pd.read_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\datasets\house_prices\train.csv') # train = load_train_house_price_from_db()
test = pd.read_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\datasets\house_prices\test.csv') # test = load_test_house_price_from_db()
# test = load_test_house_price_from_db()
print (f"Train has {train.shape[0]} rows and {train.shape[1]} columns")
print (f"Test has {test.shape[0]} rows and {test.shape[1]} columns")
train.shape, test.shape
df = pd.concat((train, test)).reset_index(drop=True) # df = train.append(test).reset_index()
df.head()
df.tail()


# General View

df.head()
df.shape
df.info()
df.columns
df.index
df.isnull().values.any()
df.isnull().sum().sort_values(ascending = False).head(15)

# Drop the columns that have many NAN values
df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis = 1, inplace = True)

# Type transformations
df.info()

# Columns to be transformed
transform_num_to_cat = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath',
                        'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                        'Fireplaces', 'GarageYrBlt', 'GarageCars', 'YrSold', 'MoSold']

# Apply dtype transformations for the selected columns
for col in transform_num_to_cat:
    df[col] = df[col].astype(str)

# Check the types again
df.info()

cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
len(cat_cols)

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in ["Id", 'SalePrice']]
len(num_cols)

# Drop 'Id' column
df.drop('Id', axis= 1, inplace=True)

# Now, we have 55 categorical and 21 numerical columns excluding 'Id' and target.


# Feature Creation


# Create a function to plot histograms
def plot_histogram(variable):
    df[variable].hist()
    plt.show()


# Create a function to plot boxplots
def plot_boxplot(variable, target, dataframe):
    sns.boxplot(x=variable, y=target, data=dataframe)
    plt.show()


df.info()

# Now, it is time to create new features to increase our model performance and improve the results

# Above, we transformed year variables from numerical to categorical variables. In order to make calsulations, we need take back the tramsformation.
df['YearBuilt'] = df['YearBuilt'].astype(int)
df['YearRemodAdd'] = df['YearRemodAdd'].astype(int)
df['YrSold'] = df['YrSold'].astype(int)

# df['YrBltAndRemod']= df['YearBuilt']+ df['YearRemodAdd']
# num_cols.append('YrBltAndRemod')

# Create a new feature that shows the age of the building
df["BldgAge"] = (df["YearBuilt"].max() - df["YearBuilt"])
# Show the histogram for 'BldgAge'
plot_histogram('BldgAge')

num_cols.append('BldgAge')

# Create a new feature that shows the age of the building when it was sold
df['BldgSoldAge'] = df['YrSold'] - df['YearBuilt']
# Show the histogram for 'BldgSoldAge'
plot_histogram('BldgSoldAge')

num_cols.append('BldgSoldAge')
cat_cols.remove('YearBuilt')
cat_cols.remove('YrSold')

# Create a new feature that shows how new was the building when it was sold
df['SoNewatSold'] = df['YrSold'] - df['YearRemodAdd']
# Show the histogram for 'SoNewatSold'
plot_histogram('SoNewatSold')


num_cols.append('SoNewatSold')
cat_cols.remove('YearRemodAdd')

# Now, let's look at 'GarageYrBlt' variable. We see there are some nan classes, which means these buildings do not have garage.
df['GarageYrBlt'].describe()
# Even though, we implemented imputaiton for missing values, there are still nan values by 'GarageYrBlt'
df.loc[df['GarageYrBlt'] == 'nan', 'GarageYrBlt'].count() # 159
df.loc[df['GarageYrBlt'] == 'nan', 'GarageYrBlt'] = 0
# In the next step we are going to create a new column called 'HasGarage' that shows if the building has a garage or not.
df['GarageYrBlt'] = df['GarageYrBlt'].astype(float)
# Create a new feature that shows the age of the garage when it was sold
df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
# # Check for missing values in the new variable
# df['GarageAge'].isnull().sum()
# # Fill missing values with 999
# df[['GarageAge']] = df[['GarageAge']].apply(lambda x: x.fillna(999))
# Show the histogram for 'GarageAge'
plot_histogram('GarageAge')

num_cols.append('GarageAge')
cat_cols.remove('GarageYrBlt')
df.drop('GarageYrBlt', axis = 1, inplace=True)

# Create a feature that shows if the house has a garage or not
df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
# Trasnform 'HasGarage' into categorical column
df['HasGarage'] = df['HasGarage'].astype(str)
# Show the correlation with 'GarageArea'
df[['HasGarage', 'GarageArea']].corr()
# Show boxplot to see difference of 'SalePrice' between the buildings with amd without garage.
plot_boxplot('HasGarage', 'SalePrice', df)

cat_cols.append('HasGarage')

# Drop the coluns, that we have used and we do no need anymore
df.drop(['YearBuilt', 'YearRemodAdd', 'YrSold', 'MoSold'], axis = 1, inplace = True)
cat_cols.remove('MoSold')

# Show te distribution of 'PoolArea'
df['PoolArea'].describe()
# Show the non_null values for 'PoolArea'
df.loc[df["PoolArea"]!=0, 'PoolArea']
# Relationship between 'PoolArea' and 'SalePrice'
sns.jointplot(x="PoolArea", y="SalePrice", data=df, kind="reg", truncate=False)
plt.show()
# We see that they are very rare, but there is a meaningful difference between the ones with and without a pool
# So, let's create a feature that shows if the house has a pool or not , and then drop 'PoolArea'
df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
# Transform 'HasPool' into categorical column
df['HasPool'] = df['HasPool'].astype(str)
# Show boxplot to see difference of 'SalePrice' between the buildings with amd without a pool.
plot_boxplot('HasPool', 'SalePrice', df)

df.drop('PoolArea', axis = 1, inplace = True)
cat_cols.append('HasPool')
num_cols.remove('PoolArea')

# Create a feature that shows if the house has the second floor or not
df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['Has2ndFloor'] = df['Has2ndFloor'].astype(str)
# Show boxplot to see difference of 'SalePrice' between the buildings with amd without a second floor.
plot_boxplot('Has2ndFloor', 'SalePrice', df)


cat_cols.append('Has2ndFloor')

# Create a feature that shows if the house has a basement or not and make the variable transformation
df['HasBsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df['HasBsmt'] = df['HasBsmt'].astype(str)
# Show boxplot to see difference of 'SalePrice' between the buildings with amd without a second floor.
plot_boxplot('HasBsmt', 'SalePrice', df)


cat_cols.append('HasBsmt')

# Create a feature that shows if the house has a fireplace or not
df['Fireplaces'] = df['Fireplaces'].astype(int)
df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df['Fireplaces'] = df['Fireplaces'].astype(str)
df['HasFireplace'] = df['HasFireplace'].astype(str)
# Show boxplot to see difference of 'SalePrice' between the buildings with amd without a second floor.
plot_boxplot('HasFireplace', 'SalePrice', df)


cat_cols.append('HasFireplace')

# Create a feature that shows the total number of bathrooms
# We need to transform the variables to float data type, to be able to make calculations.
for col in ['HalfBath', 'BsmtHalfBath', 'BsmtFullBath', 'FullBath']:
    df[col] = df[col].astype(float)
# Create a feature that shows the total number of bathrooms
df["TotBath"] = ((0.5 * df["HalfBath"]) + (0.5 * df["BsmtHalfBath"]) + (df["BsmtFullBath"]) + (df["FullBath"]))
# See the distribution of 'TotBath'
df["TotBath"].describe()
# Relationship between 'TotBath' and 'SalePrice'
sns.jointplot(x="TotBath", y="SalePrice", data=df, kind="reg", truncate=False)
plt.show()

num_cols.append('TotBath')

drop_list_bath = ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"]
cat_cols = [col for col in cat_cols if col not in drop_list_bath]
len(cat_cols)
# Drop other bathroom columns
for col in drop_list_bath:
    df.drop(col, axis=1, inplace=True)

# Create a new feature that shows if the the surface of the building
df['TotalSF'] = (df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'])
# Show the correlations
df[['TotalSF', 'TotalBsmtSF', '1stFlrSF','2ndFlrSF']].corr()
# Relationship between 'TotBath' and 'SalePrice'
sns.jointplot(x="TotalSF", y="SalePrice", data=df, kind="reg", truncate=False)
plt.show()

# For now do not drop the columns we used to create a new variable
# # df.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'], axis = 1, inplace=True)

num_cols.append('TotalSF')
# # num_cols.remove('TotalBsmtSF')
# # num_cols.remove('1stFlrSF')
# # num_cols.remove('2ndFlrSF')

# Create a feature that shows the total surface of porchs
df["TotPorchSF"] = (df["ScreenPorch"] + df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["WoodDeckSF"])
# Show the correlations
df[['TotPorchSF', 'ScreenPorch', 'OpenPorchSF','EnclosedPorch', '3SsnPorch', 'WoodDeckSF', 'SalePrice']].corr()
# Relationship between 'TotPorchSF' and 'SalePrice'
sns.jointplot(x="TotPorchSF", y="SalePrice", data=df, kind="reg", truncate=False)
plt.show()

num_cols.append('TotPorchSF')

# Create a feature that shows if the building has masonry area or not
df['MasVnrArea' + 'HasOrNot'] = np.where(df['MasVnrArea'] > 0, 1, 0)
df['MasVnrAreaHasOrNot'] = df['MasVnrAreaHasOrNot'].astype(str)
# Show boxplot to see difference of 'SalePrice' between the buildings with and without a masonry area.
plot_boxplot('MasVnrAreaHasOrNot', 'SalePrice', df)

cat_cols.append('MasVnrAreaHasOrNot')


df.info()

print('Now, we have {} categorical and {} numerical variables'.format(len(cat_cols), len(num_cols)))

# Now, we have 52 categorical and 25 numerical variables


# Outlier Analysis


# Function to calculate outlier thresholds
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Function to report variables with outliers and return the names of the variables with outliers with a list
def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names


has_outliers(df, num_cols)


# Function to reassign up/low limits to the ones above/below up/low limits
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Function to reassign up/low limits to the ones above/below up/low limits by using apply and lambda method
def replace_with_thresholds_with_lambda(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].apply(lambda x: up_limit if x > up_limit else (low_limit if x < low_limit else x))


# Assign outliers thresholds values for all the numerical variables
for col in num_cols:
    replace_with_thresholds(df, col)

# Check for outliers, again
has_outliers(df, num_cols)


# Missing Values Analysis


# Function to catch missing variables, count them and find ratio (in descending order)
def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na


missing_values_table(df)


# Numerical variables

# Check for missing values
df[num_cols].isnull().sum()
df[num_cols].isnull().sum().sum() # 517

# Impute median values for missing values for numeral variables
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()), axis=0)

# Check for missing values, again and control
df[num_cols].isnull().sum()
df[num_cols].isnull().sum().sum() # 0


# Categorical variables

# Check for missing values
df[cat_cols].isnull().sum()
df[cat_cols].isnull().sum().sum() # 1075

# Fill missing values for the buildings with these variables with 0, because they do not have any garage.
for col in ('GarageArea', 'GarageCars'):
    df[col] = df[col].fillna(0)

# Some missing values are intentionally left blank, for example: In the Alley feature there are blank values meaning that there are no alley's in that specific house.
missing_val_col = ["GarageType", "GarageFinish", "GarageQual", "GarageCond", 'BsmtQual',
                   'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

# Fill these missing values with 'None'
for i in missing_val_col:
    df[i] = df[i].fillna('None')

# Fill other missing values with the most logical and common classes.

# For 'MSZoning' fill the missing values with respect to 'MSSubClass' classes
df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

# For the following variables fill the missing values with the most common classes/modes
cat_cols_to_fill = ['Functional', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType', 'SaleType', 'Electrical']

for col in cat_cols_to_fill:
    df[col] = df[col].fillna(df[col].mode()[0])

# Check for missing values, again control
df[cat_cols].isnull().sum()
df[cat_cols].isnull().sum().sum() # 0

missing_values_table(df)
# Now, we have missing values only for tagrge variable, which we will not fill.


# Label and One Hot Encoding


# Define a function to apply one hot encoding to categorical variables.
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns


df, new_cols_ohe = one_hot_encoder(df, cat_cols)
df.head()
len(new_cols_ohe)

# Check if there are any duplicate_columns
duplicate_columns = df.columns[df.columns.duplicated()]
df = df.loc[:, ~df.columns.duplicated()]

# # Analyse categorical variables, again.
# cat_summary(df, new_cols_ohe, "SalePrice")


# Standardization

df.head()

# MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
transformer = MinMaxScaler()
df[num_cols] = transformer.fit_transform(df[num_cols])  # default value is between 0 and 1

df[num_cols].describe().T
len(num_cols)


# Check before modeling for missing values and outliers in the dataset
missing_values_table(df)
has_outliers(df, num_cols)

df.isnull().sum().sort_values(ascending=False)
df.isnull().sum().sum()

# Last look at the dataset
df.head()
df.info()

# Pickle the data set after preprocessing and feature engineering to be able avoid repetations.

# At the beginning, we gathered the train and test datasets to implement the same processes for the dataset.
# Now, it is time separate them.

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

train_df.to_pickle("datasets/house_prices/prepared_data/train_df_.pkl")
test_df.to_pickle("datasets/house_prices/prepared_data/test_df_.pkl")
