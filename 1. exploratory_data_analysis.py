# HOUSE PRICE PREDICTION MODEL

# EXPLORATORY DATA ANALYSIS

# Import dependencies
import pandas as pd
import numpy as np
import pymysql as pymysql
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
from sklearn.exceptions import ConvergenceWarning
import missingno as msno
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Load the dataset and gather train and test sets
train = pd.read_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\datasets\house_prices\train.csv') # train = load_train_house_price_from_db()
test = pd.read_csv(r'C:\Users\yakup\PycharmProjects\dsmlbc\datasets\house_prices\test.csv') # test = load_test_house_price_from_db()
# from dsmlbc_functions import load_house_price
# train_df, test_df = load_house_price()
print (f"Train has {train.shape[0]} rows and {train.shape[1]} columns")
print (f"Test has {test.shape[0]} rows and {test.shape[1]} columns")
train.shape, test.shape
df = pd.concat((train, test)).reset_index(drop=True) # df = train.append(test).reset_index()
df.head()
df.tail()


# 1. General View

df.head()
df.shape
df.info()
df.columns
df.index
df.isnull().values.any()
df.isnull().sum().sort_values(ascending = False).head(15)

# Visualize missing variables
sns.heatmap(df.isnull(), cbar=False, cmap='magma')
plt.show()
# Missing values overall view
msno.bar(df)
plt.show()
# Now, we can see the relationship between missing values
msno.matrix(df)
plt.show()
# Nullity correlation visualization
msno.heatmap(df)
plt.show()

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

# Now, we have 55 categorical and 19 numerical columns excluding 'Id' and target.


# 2. Categorical Variables Analysis

# Bring categorical variables
cat_cols

# cat_summary function. extra feature --> it describes target, too.
def cat_summary(data, categorical_cols, target, number_of_classes=30, plot=False):
    var_count = 0  # How many categorical variables will be reported
    vars_more_classes = []  # Variables with more than a certain number of classes will be stored.
    for var in categorical_cols:
        if len(data[var].value_counts()) <= number_of_classes:  # choose by number of classes
            print(pd.DataFrame({var: data[var].value_counts(),
                                "Ratio": 100 * data[var].value_counts() / len(data),
                                "TARGET_MEAN": data.groupby(var)[target].mean(),
                                "TARGET_MEDIAN": data.groupby(var)[target].median()}), end="\n\n\n")
            var_count += 1
            if plot:
                sns.boxplot(x=var, y=target, data=data)
                plt.show()
        else:
            vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


cat_summary(df, cat_cols, "SalePrice", plot=True)


# Analyse variables with more than 10 classes
for col in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    print(df[col].value_counts())

# Drop the columns that are not important for us


# 3. Numerical Variables Analysis

# Bring numerical variables
num_cols


# Function to plot histograms for numerical variables
def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")


hist_for_nums(df, num_cols)


# 4. Target Analysis

# See the distribution of the target value with respect to quantiles
df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])

# Visualize the target variable
df.SalePrice.hist()
plt.show()

np.log1p(df.SalePrice).hist()
plt.show()


# Correlations of target with numerical independent variables by using a correlation limit 0.60
def find_correlation(dataframe, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in num_cols:
        if col == "SalePrice":
            pass

        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df)


# Visualize correlations by using heatmap.

# Plot fig sizing.
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize=(30,20))
# # Generate a mask for the upper triangle (taken from seaborn example gallery)
# mask = np.zeros_like(train.corr(), dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# Plotting heatmap
sns.heatmap(train.corr(), cmap=sns.diverging_palette(20, 220, n=200), annot=True, center = 0) # mask=mask
plt.title("Heatmap of all the Features", fontsize = 30)
plt.show()

# Correlation matrix with highest 10 variables
k = 10 #number of variables for heatmap
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
corr_cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[corr_cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=corr_cols.values, xticklabels=corr_cols.values)
plt.show()

# Scatterplot
sns.set()
corr_cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[corr_cols], size = 2.5)
plt.show()

# Anaother way to do it.
corr_new_train=train.corr()
plt.figure(figsize=(5,15))
sns.heatmap(corr_new_train[['SalePrice']].sort_values(by=['SalePrice'],ascending=False).head(30),annot_kws={"size": 16},vmin=-1, cmap='PiYG', annot=True)
sns.set(font_scale=2)
plt.show()