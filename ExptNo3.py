import pandas as pd
import numpy as np

df = pd.read_csv('weight-height.csv')

df.columns = df.columns.str.strip()


# Calculating Z-Score
df['Height_Zscore'] = np.abs(
    (df['Height'] - df['Height'].mean()) / df['Height'].std())
df['Weight_Zscore'] = np.abs(
    (df['Weight'] - df['Weight'].mean()) / df['Weight'].std())

threshold = 3

outliers_height = df[df['Height_Zscore'] > threshold]
outliers_weight = df[df['Weight_Zscore'] > threshold]

print("Outliers for height:")
print(outliers_height)

print("\nOutliers for weight:")
print(outliers_weight)
