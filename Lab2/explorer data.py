import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 1. Load data
train_file_path = "training_set_pixel_size_and_HC.csv"
test_file_path = "test_set_pixel_size.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# 2. Explore data
print("Dataset Information:")
print(train_df.info())
print("\nStatistical Summary:")
print(train_df.describe())

# Check for missing values
missing_values = train_df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Check for duplicate data
duplicated_rows = train_df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicated_rows}")

# Check data distribution
plt.figure(figsize=(8, 5))
sns.histplot(train_df["pixel size(mm)"], kde=True, bins=30)
plt.title("Distribution of Pixel Size")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(train_df["head circumference (mm)"], kde=True, bins=30)
plt.title("Distribution of Head Circumference")
plt.show()

# 3. Visualize the relationship between Pixel Size and Head Circumference
plt.figure(figsize=(8, 5))
sns.scatterplot(x=train_df["pixel size(mm)"], y=train_df["head circumference (mm)"])
plt.xlabel("Pixel Size (mm)")
plt.ylabel("Head Circumference (mm)")
plt.title("Relationship between Pixel Size and Head Circumference")
plt.show()