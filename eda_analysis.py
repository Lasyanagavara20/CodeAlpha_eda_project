import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Optional: suppress layout warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load data
df = pd.read_csv("titanic.csv")
print(df.head())

# Basic structure
print("Shape of the dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types and non-null counts:")
print(df.info())
print("\nSummary statistics for numeric columns:")
print(df.describe())

# Separate features
numerical = df.select_dtypes(include=['int64', 'float64']).columns
categorical = df.select_dtypes(include=['object']).columns

print("Numerical Columns:", list(numerical))
print("Categorical Columns:", list(categorical))

# Check for missing values
print("Missing values per column:\n")
print(df.isnull().sum())

# Spot distributions and outliers
fig = plt.figure(figsize=(14, 10))
df[numerical].hist(bins=20, layout=(3, 3), figsize=(14, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("spot_distributions_outliers.png")
plt.show()

# Visualize categorical features
for col in categorical:
    print(f"\nValue counts for {col}:\n", df[col].value_counts())
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, data=df)
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"countplot_{col.lower()}.png")  # Unique file per column
    plt.show()

# Explore relationships between variables
plt.figure(figsize=(10, 6))
sns.heatmap(df[numerical].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

# Survival by sex
plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Sex")
plt.tight_layout()
plt.savefig("survival_by_sex.png")
plt.show()

# Survival by passenger class
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.tight_layout()
plt.savefig("survival_by_class.png")
plt.show()

# Boxplots for outliers
for col in ['Age', 'Fare']:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(f"boxplot_{col.lower()}.png")  # Unique file per variable
    plt.show()







