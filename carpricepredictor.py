# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import KNNImputer

# %%
df = pd.read_csv('/kaggle/input/car-price-prediction-challenge/car_price_prediction.csv')

numerical_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)
categorical_cols = [col for col in list(df.columns) if col not in numerical_cols]

# %%
df = df[~df.duplicated()]

# %%
df.columns

# %%
df[numerical_cols].head()

# %%
df[categorical_cols].head()

# %%
categorical_cols

# %%
df.isnull().sum()

# %%
# Converting Levy from categorical to numerical

df['Levy'] = df['Levy'].replace(r'[^\d.]', np.nan, regex=True)
df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')

numerical_cols.append('Levy')
categorical_cols = [col for col in categorical_cols if col != 'Levy']

# %%
# Converting Leather Interior from categorical to numerical

df['Leather interior'] = df['Leather interior'].map({
    'Yes':1,
    'No':0
})

numerical_cols.append('Leather interior')
categorical_cols = [col for col in categorical_cols if col != 'Leather interior']

# %%
# Converting Engine Volume from categorical to numerical

df['Turbo'] = df['Engine volume'].str.contains('Turbo', case=False, na=False).astype(int)
df['Engine volume'] = df['Engine volume'].str.replace(' Turbo', '', regex=False )
df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce')

numerical_cols.append('Engine volume')
numerical_cols.append('Turbo')
categorical_cols = [col for col in categorical_cols if col != 'Engine volume']

# %%
# Converting Mileage from categorical to numerical

df['Mileage'] = df['Mileage'].str.replace(' km', '', regex=False)
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')

numerical_cols.append('Mileage')
categorical_cols = [col for col in categorical_cols if col != 'Mileage']

# %%
df = df[numerical_cols]
df.isnull().sum()

# %%
imputer = KNNImputer(n_neighbors=2)
imputed_data = imputer.fit_transform(df)
imputed_df = pd.DataFrame(imputed_data, columns=numerical_cols)
df = imputed_df

# %%
numerical_cols = [col for col in numerical_cols if col != 'ID']
df = df.drop('ID', axis=1)
df[numerical_cols].describe()

# %%
df = df[df['Price'] < (1*10**7)]

# %%
iqr = 1.5 * 118749
upper_fence = 188888 + iqr
lower_fence = 70139 - iqr

df = df[(df['Mileage'] >= lower_fence) & (df['Mileage'] <= upper_fence)]

# %%
df['Mileage'].skew()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Mileage', y='Price', data=df)
plt.title('Price per Mileage')
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(df['Mileage'], bins=20, kde=True)
plt.title('Histogram of Mileage')
plt.xlabel('Mileage')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='Mileage', data=df)
plt.title('Boxplot of Mileage')
plt.xlabel('Mileage')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# %%
df['Prod. year'].skew()

# %%
df['Car Age'] = 2020 - df['Prod. year']
df = df.drop('Prod. year', axis=1)
df['Car Age'] = np.sqrt(df['Car Age'])
df['Car Age'].skew()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Car Age', y='Price', data=df)
plt.title('Price per Age')
plt.xlabel('Age')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(df['Car Age'], bins=30, kde=True)
plt.title('Histogram of Car Age')
plt.xlabel('Car Age')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# %%
df = df[df['Engine volume'] <= 19]

# %%
df['Engine volume'].skew()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Engine volume', y='Price', data=df)
plt.title('Price against Engine volume')
plt.xlabel('Engine volume')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(df['Engine volume'], bins=20, kde=True)
plt.title('Histogram of Engine volume')
plt.xlabel('Engine volume')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# %%
df['Cylinders'].skew()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Cylinders', y='Price', data=df)
plt.title('Price against Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='Cylinders', data=df)
plt.title('Countplot for Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# %%
df['Airbags'].skew()

# %%
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Airbags', y='Price', data=df)
plt.title('Price against Airbags')
plt.xlabel('Airbags')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.countplot(x='Airbags', data=df)
plt.title('Countplot for Airbags')
plt.xlabel('Airbags')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# %%
df['Price'].skew()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(df['Price'], bins=20, kde=True)
plt.title('Histogram for Price')
plt.xlabel('Price')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(x='Price', data=df)
plt.title('Boxplot for Price')
plt.xlabel('Price')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# %%
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', fmt='.2f', linewidths=.5, annot=True)
plt.show()

# %%
high_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) >= 0.25:
            feature1 = correlation_matrix.columns[i]
            feature2 = correlation_matrix.columns[j]
            correlation = correlation_matrix.iloc[i, j]
            high_correlations.append({
                'Feature 1':feature1,
                'Feature 2':feature2,
                'Correlation':correlation
            })

high_corr_df = pd.DataFrame(high_correlations)
high_corr_df

# %%
df.head()

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: ${rmse:,.2f}")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R² Score: {r2:.3f}")
