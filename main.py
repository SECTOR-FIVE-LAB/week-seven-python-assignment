import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset from sklearn and convert to DataFrame
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("First 5 rows of the dataset:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())

    # No missing values in iris dataset; otherwise, we would do:
    # df.fillna(method='ffill', inplace=True)

except FileNotFoundError:
    print("Dataset file not found.")
except Exception as e:
    print("An error occurred:", e)

# Task 2: Basic Data Analysis
print("\nBasic statistics:")
print(df.describe())

print("\nMean measurements per species:")
print(df.groupby('species').mean())

# Task 3: Data Visualization
sns.set(style="whitegrid")

# 1. Line chart - For this example, simulate a trend
df['index'] = df.index
plt.figure(figsize=(10, 5))
plt.plot(df['index'], df['sepal length (cm)'], label='Sepal Length')
plt.title('Trend of Sepal Length over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar chart - Average petal length per species
plt.figure(figsize=(8, 5))
df.groupby('species')['petal length (cm)'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram - Distribution of sepal width
plt.figure(figsize=(8, 5))
plt.hist(df['sepal width (cm)'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot - Sepal length vs. petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs. Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()

# Findings:
print("\nObservations:")
print("- Setosa species has notably shorter petal lengths compared to others.")
print("- There is a visible correlation between sepal and petal length, especially in versicolor and virginica.")
print("- Sepal width appears to be normally distributed around 3.0 cm.")
