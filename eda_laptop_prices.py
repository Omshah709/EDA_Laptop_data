import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('laptop_prices.csv')
print(df) 
print(df.isnull().sum())  # Check for missing values
import re
print(df.columns) #Before Cleaning  column names
# Function to clean column names
def clean_column_names(df):
    df.columns = (df.columns
                  .str.lower()  # Convert to lowercase
                  .str.strip()  # Remove extra spaces
                  .str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters
                  .str.replace(r'\s+', '_', regex=True))  # Replace spaces with underscores
    return df

# Apply function
df = clean_column_names(df)

# Check updated column names
print(df.columns)


print(df.info())  # Check column data types and missing values
print(df.shape)   # Rows & columns count
print(df.head())  # Preview first 5 rows

df.drop_duplicates(inplace=True)
print("Duplicates removed:", df.shape) 

print(df.describe())  # Summary of numerical columns
print(df.describe(include='object'))  # Summary of categorical columns


import seaborn as sns
import matplotlib.pyplot as plt

# Remove extreme outliers using IQR method
Q1 = df['price_'].quantile(0.25)
Q3 = df['price_'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price_'] >= Q1 - 1.5 * IQR) & (df['price_'] <= Q3 + 1.5 * IQR)]

# Price Boxplot
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='price_')
plt.title("Price Distribution (Boxplot)")
plt.show()

# Univariate Analysis
# Price Distribution graph
plt.figure(figsize=(10,5))
sns.histplot(df['price_'], bins=30, kde=True)
plt.title("Laptop Price Distribution")
plt.xlabel("Price ($)")
plt.show()

# Popularity graph
plt.figure(figsize=(12,5))
sns.countplot(data=df, x='brand', order=df['brand'].value_counts().index)
plt.xticks(rotation=45)
plt.title("Laptop Brand Popularity")
plt.show()

# Ram Distribution Graph
sns.histplot(df['ram_gb'], bins=10, kde=True)
plt.title("RAM Distribution")
plt.show()

# Storage capacity Distribution Graph
sns.histplot(df['storage'], bins=10, kde=True)
plt.title("Storage Capacity Distribution")
plt.show()

import matplotlib.pyplot as plt

# Count the number of laptops per brand
brand_counts = df['brand'].value_counts()

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(brand_counts, labels=brand_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors, startangle=140)
plt.title("Laptop Brand Distribution")
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
plt.show()

#Bivariate Analysis
# Price VS Ram
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='ram_gb', y='price_')
plt.title("Price vs RAM")
plt.show()

# Price VS Processor
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='processor', y='price_')
plt.xticks(rotation=45)
plt.title("Price vs Processor")
plt.show()

# Battery Life vs Laptop Weight
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='weight_kg', y='battery_life_hours')
plt.title("Battery Life vs Laptop Weight")
plt.show()


# Selecting only numerical columns
num_df = df.select_dtypes(include=['number'])

# Correlation Matrix
plt.figure(figsize=(10,6))
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Numerical Features Only)")
plt.show()
