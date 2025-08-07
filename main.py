import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Set plot style
sns.set(style="whitegrid")

# ============================
# 1. Survival Count by Gender
# ============================
plt.figure(figsize=(8, 5))
sns.countplot(x='sex', hue='survived', data=df, palette='Set2')
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(labels=['Not Survived', 'Survived'])
plt.tight_layout()
plt.show()

# ============================
# 2. Age Distribution
# ============================
plt.figure(figsize=(8, 5))
sns.histplot(df['age'].dropna(), bins=30, kde=True, color='skyblue')
plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================
# 3. Survival Rate by Class
# ============================
plt.figure(figsize=(8, 5))
sns.barplot(x='pclass', y='survived', data=df, palette='pastel')
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# ============================
# 4. Heatmap of Correlations
# ============================
plt.figure(figsize=(10, 6))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.show()

# ============================
# 5. Violin Plot - Age vs Class
# ============================
plt.figure(figsize=(8, 5))
sns.violinplot(x='pclass', y='age', data=df, palette='muted')
plt.title("Distribution of Age by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.tight_layout()
plt.show()
