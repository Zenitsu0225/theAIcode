import pandas as p
from sklearn.utils import resample

data = p.read_csv('dataset.csv')

minority_class = data[data['stroke'] == 1]
majority_class = data[data['stroke'] == 0]


print("Before sampling")

print(minority_class['stroke'].value_counts())
print(majority_class['stroke'].value_counts())

minority_upsampled = resample(minority_class,  # Resampling minority class
                              replace=True,
                              n_samples=len(majority_class),
                              random_state=42)

balanced_data = p.concat([majority_class, minority_upsampled])

print("\nAfter samping")

print(balanced_data['stroke'].value_counts())

balanced_data.to_csv('balanced_dataset.csv', index=False)
