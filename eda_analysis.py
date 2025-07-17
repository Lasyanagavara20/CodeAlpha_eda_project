import pandas as pd

df = pd.read_csv("titanic.csv")
print(df.head())
# Basic structure
print("Shape of the dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types and non-null counts:")
print(df.info())
print("\nSummary statistics for numeric columns:")
print(df.describe())
#Separate features to explore them properly
numerical = df.select_dtypes(include=['int64', 'float64']).columns
categorical = df.select_dtypes(include=['object']).columns

print("Numerical Columns:", list(numerical))
print("Categorical Columns:", list(categorical))
#Check for missing values
print("Missing values per column:\n")
print(df.isnull().sum())
#Spot distributions and outliers
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(14, 10))
df[numerical].hist(bins=20, layout=(3, 3), figsize=(14, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("spot_distributions_outliers.png")
plt.show()
#Visualize categorical features
for col in categorical:
    print(f"\nValue counts for {col}:\n", df[col].value_counts())
    sns.countplot(x=col, data=df)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("categorical_features.png")
    plt.show()
#Explore relationships between variables
plt.figure(figsize=(10, 6))
sns.heatmap(df[numerical].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("Correlation_Matrix.png")
plt.show()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Sex")
plt.savefig("Survival_by_Sex.png")
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.savefig("Survival_by_Passenger_Class.png")
plt.show()
#boxplot to detect outliers
for col in ['Age', 'Fare']:
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig("boxplot_to_detect_outliers.png")
    plt.show()