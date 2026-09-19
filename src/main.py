"""
House Prices Prediction — Kaggle Top 1% Solution
An end-to-end machine learning project that predicts residential home prices using comprehensive feature engineering, statistical outlier treatment, and tuned gradient boosting models to achieve a Top 1% ranking on Kaggle.

Original Kaggle Notebook: https://www.kaggle.com/code/lazer999/top-1-housing-price-eda-random-for-everyone
Author: Muhammad Musa Khan (Kaggle Master: https://kaggle.com/lazer999)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# --- Smart Dataset Path Resolution ---
def _resolve_data_path(file_path):
    """Checks local and data/ directories if dataset path is missing."""
    if os.path.exists(file_path):
        return file_path
    base = os.path.basename(file_path)
    candidates = [
        base,
        os.path.join("data", base),
        os.path.join("..", "data", base),
        file_path.replace("/kaggle/input/", "data/"),
        file_path.replace("../input/", "data/"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return file_path

# --- Pipeline Execution ---

# --- Cell 1 ---
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# --- Cell 2 ---
df_train = pd.read_csv('../input/home-data-for-ml-course/train.csv')
df_test = pd.read_csv('../input/home-data-for-ml-course/test.csv')

# --- Cell 3 ---
df_train.head()

# --- Cell 4 ---
r,c = df_train.shape
print('The training data has {} rows and {} columns'.format(r,c))
r,c = df_test.shape
print('The validation data has {} rows and {} columns'.format(r,c))

# --- Cell 5 ---
df_train.info()

# --- Cell 6 ---
plt.figure(figsize=(24,8))
# columns with the most null values
cols_with_null=df_train.isnull().sum().sort_values(ascending=False)
# Let's visualize these columns
sns.barplot(x=cols_with_null.index,y=cols_with_null)
plt.xticks(rotation=90)
plt.show;

# --- Cell 7 ---
cols_with_null.head(10)

# --- Cell 9 ---
df_train['SalePrice'].isnull().sum()

# --- Cell 10 ---
df_train.head()

# --- Cell 11 ---
df_train.describe()

# --- Cell 12 ---
important_features=['YearBuilt','LotArea','OverallQual','OverallCond','GrLivArea','1stFlrSF','2ndFlrSF','BedroomAbvGr','OpenPorchSF','PoolArea','SalePrice']
df_train[important_features].describe()

# --- Cell 13 ---
plt.figure(figsize=(15,12))
sns.heatmap(df_train.corr())
plt.show()

# --- Cell 14 ---
un_imp=['MSSubClass','OverallCond','BsmtFinSF2','LowQualFinSF','BsmtHalfBath','3SsnPorch','YrSold','MoSold','MiscVal','PoolArea']

# --- Cell 16 ---
fig, ax=plt.subplots(1,3,figsize=(28,7))
sns.scatterplot(x=df_train.GrLivArea,y=df_train.SalePrice,size=df_train.BedroomAbvGr,hue=df_train.OverallQual, ax=ax[0])
ax[0].set_title("Ground Living Area")
sns.scatterplot(x=df_train.LotArea,y=df_train.SalePrice,size=df_train.BedroomAbvGr,hue=df_train.OverallQual, ax=ax[1])
ax[1].set_title("LOT AREA")
sns.boxplot(x=df_train.SalePrice);

# --- Cell 17 ---
sns.catplot(data=df_train, y='SalePrice', x='OverallQual', kind="boxen"); #mutli col bar plot.

# --- Cell 18 ---
df_train['SalePrice'].quantile(0.995)

# --- Cell 19 ---
rows_2_drop=df_train[df_train['SalePrice']>df_train['SalePrice'].quantile(0.995)].index
df_train.drop(rows_2_drop,inplace=True)

# --- Cell 20 ---
df_train.shape

# --- Cell 21 ---
rows_2_drop=df_train[df_train['GrLivArea']>4000].index
df_train.drop(rows_2_drop,inplace=True)
df_train.shape

# --- Cell 22 ---
df_train[df_train['LotArea']>100000]

# --- Cell 23 ---
rows_2_drop=df_train[df_train['LotArea']>100000].index
df_train.drop(rows_2_drop,inplace=True)
df_train.shape

# --- Cell 24 ---
X_train = df_train.drop(['Id','SalePrice'],axis=1)
y_train = df_train.SalePrice
X_test = df_test.drop(['Id'],axis=1)

# --- Cell 25 ---
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

# --- Cell 26 ---
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])

# --- Cell 27 ---
from sklearn.ensemble import RandomForestClassifier
model_GBR =  GradientBoostingRegressor(n_estimators=1100, loss='squared_error', subsample = 0.35, learning_rate = 0.05,random_state=1)
GBR_Pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model_GBR)])
GBR_Pipeline.fit(X_train, y_train)
preds_GBR = GBR_Pipeline.predict(X_test)

# --- Cell 28 ---
submission= pd.DataFrame({'Id': df_test.Id,'SalePrice': preds_GBR})


# --- Cell 29 ---
submission.head()

# --- Cell 30 ---
submission.to_csv('submission.csv',index=False)

# --- Cell 31 ---
516/49003



if __name__ == "__main__":
    print("Pipeline execution complete.")
