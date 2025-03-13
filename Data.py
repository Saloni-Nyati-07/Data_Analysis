import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset as an example
df = sns.load_dataset("titanic")
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Drop rows with too many missing values
df.dropna(subset=['age', 'embark_town'], inplace=True)

# Fill missing values in numerical columns with the median
df['age'].fillna(df['age'].median(), inplace=True)

# Fill missing values in categorical columns with the mode
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

# Check for duplicates
print("Duplicates:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df['age'])
plt.show()
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['age'] >= lower_bound) & (df['age'] <= upper_bound)]

# Convert categorical values to lowercase
df['embark_town'] = df['embark_town'].str.lower()

# Fix common typos (Example: 'southampton' has variations)
df['embark_town'] = df['embark_town'].replace({'southampoton': 'southampton'})

# Summary statistics for numerical columns
print(df.describe())

# Mode for categorical variables
print(df.mode().iloc[0])

# Count of unique values in categorical columns
print(df['embark_town'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x=df['embark_town'])
plt.title("Distribution of Embark Towns")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(5, 5))
sns.boxplot(y=df['age'])
plt.title("Boxplot of Age")
plt.show()

# Compute correlation matrix
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix)

# Heatmap to visualize correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#Age vs.Fare
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['age'], y=df['fare'])
plt.title("Scatter Plot: Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# Which embark town had higher average fares?
plt.figure(figsize=(6, 4))
sns.barplot(x=df['embark_town'], y=df['fare'])
plt.title("Average Fare by Embark Town")
plt.show()

plt.figure(figsize=(6, 4))
sns.violinplot(x=df['class'], y=df['age'])
plt.title("Age Distribution by Passenger Class")
plt.show()

#Shows outliers in fare prices for different classes.
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['class'], y=df['fare'])
plt.title("Fare Distribution by Class")
plt.show()

# Pair plot for numerical columns
sns.pairplot(df, diag_kind="kde")
plt.show()

# Compute correlation matrix
correlation_matrix = df.corr(numeric_only=True)

# Heatmap visualization
#Identify strong relationships (red for positive, blue for negative correlations).
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

#Compares survival rates across different passenger classes for males vs. females.
plt.figure(figsize=(6, 4))
sns.barplot(x="class", y="survived", hue="sex", data=df)
plt.title("Survival Rate by Class and Gender")
plt.show()

#Shows how fares vary across classes and embarkation points.
plt.figure(figsize=(8, 5))
sns.boxplot(x="class", y="fare", hue="embark_town", data=df)
plt.title("Fare Distribution by Class and Embark Town")
plt.show()