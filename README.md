**EXP 3 - Delhi Air Quality Analysis**

**Aim**


To compare air quality parameters in Delhi across different stations and analyze the relationship between pollutants (e.g., PM2.5 and NO₂) using scatter plots and correlation analysis.


**Procedure / Algorithm**

1)Load the dataset using pandas.

2)Preprocess the data:

3)Convert the date column (period.datetimeFrom.utc) to datetime format.

4)Drop missing or invalid values.

5)Pivot the dataset so each pollutant (parameter) becomes a separate column.

6)Plot scatter plot between PM2.5 and NO₂ to study their relationship.

7)Plot correlation heatmap between all pollutants to identify relationships.

8)Interpret the results — identify which pollutants are correlated and which stations are most polluted.


**Program**

**Name : Sowmiya R**

**Reg No: 212225040420**


# Q1. Load the dataset and perform preprocessing
```
## Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

## Load dataset
df = pd.read_csv("del-sirifort-cpcb-2024-25.csv")

## Display dataset information
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

## Select required columns
columns = [
    'PM2.5 (µg/m³)',
    'PM10 (µg/m³)',
    'NO2 (µg/m³)',
    'Ozone (µg/m³)'
]

data = df[columns].copy()

## Convert selected columns into numeric format
for col in columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

## Check missing values
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

## Handle missing values separately for each variable
clean_data = data.copy()

print("\nAvailable Records After Preprocessing:")
for col in columns:
    print(col, ":", clean_data[col].count())

print("\nDataset Information:")
print(clean_data.info())

print("\nPreprocessed Dataset:")
print(clean_data.head())
```

# Q2. Distribution analysis using histograms
```
plt.figure(figsize=(14, 10))
for i, col in enumerate(columns, 1):

    plt.subplot(2, 2, i)

    sns.histplot(
        clean_data[col].dropna(),
        bins=30,
        kde=True
    )

    plt.title("Distribution of " + col)
    plt.xlabel("Concentration (µg/m³)")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
```

# Q3. Calculate numerical summary
```
summary = pd.DataFrame(index=columns)

for col in columns:

    x = clean_data[col].dropna()
    summary.loc[col, 'Mean'] = x.mean()
    summary.loc[col, 'Median'] = x.median()
    summary.loc[col, 'Mode'] = x.mode().iloc[0]
    summary.loc[col, 'Minimum'] = x.min()
    summary.loc[col, 'Q1'] = x.quantile(0.25)
    summary.loc[col, 'Q3'] = x.quantile(0.75)
    summary.loc[col, 'Maximum'] = x.max()

print("NUMERICAL SUMMARY")
print(summary.round(2))
```
# Q4. Calculate measures of spread
```
spread = pd.DataFrame(index=columns)

for col in columns:

    x = clean_data[col].dropna()

    mean = x.mean()

    # Range
    spread.loc[col, 'Range'] = x.max() - x.min()

    # Variance
    spread.loc[col, 'Variance'] = x.var()

    # Standard Deviation
    spread.loc[col, 'Standard Deviation'] = x.std()

    # Coefficient of Variation
    spread.loc[col, 'Coefficient of Variation (%)'] = (
        (x.std() / mean) * 100 if mean != 0 else np.nan
    )

print("MEASURES OF SPREAD")
print(spread.round(2))
```

# Q5. Construct box plots and detect outliers using IQR
# Box plot visualization
```
plt.figure(figsize=(12, 7))

sns.boxplot(data=clean_data)

plt.title("Box Plot of Delhi Air Quality Pollutants")
plt.xlabel("Air Pollutants")
plt.ylabel("Concentration (µg/m³)")

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
```

# Outlier detection using IQR
```
for col in columns:

    x = clean_data[col].dropna()

    # Calculate quartiles
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Calculate lower and upper limits
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = x[
        (x < lower_limit) | (x > upper_limit)
    ]

    print("\nPollutant:", col)
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Number of Outliers:", len(outliers))

    print("First 10 Outliers:")
    print(outliers.head(10).values)
```

# Q6. Perform Min-Max scaling
```
# Create an empty DataFrame
minmax_data = pd.DataFrame(index=clean_data.index)

for col in columns:

    x = clean_data[col]

    min_value = x.min()
    max_value = x.max()

    # Apply Min-Max scaling
    if max_value != min_value:
        minmax_data[col] = (
            (x - min_value) / (max_value - min_value)
        )
    else:
        minmax_data[col] = 0.0

print("MIN-MAX SCALED DATA")
print(minmax_data.head(10))

# Display minimum and maximum after scaling
print("\nScaled Data Range:")
print(minmax_data.agg(['min', 'max']).round(4))

# Save complete scaled dataset
minmax_data.to_csv("minmax_scaled_data.csv", index=False)
```

# Q7. Perform Z-score standardization using StandardScaler
```
# Create an empty DataFrame
standardized_data = pd.DataFrame(index=clean_data.index)

# Apply StandardScaler separately to each pollutant
for col in columns:

    valid = clean_data[col].notna()

    scaler = StandardScaler()

    standardized_data.loc[valid, col] = (
        scaler.fit_transform(
            clean_data.loc[valid, [col]]
        ).flatten()
    )

print("Z-SCORE STANDARDIZED DATA")
print(standardized_data.head(10))

# Verify mean and standard deviation
print("\nMean After Standardization:")
print(standardized_data.mean().round(4))

print("\nStandard Deviation After Standardization:")
print(standardized_data.std(ddof=0).round(4))

# Save complete standardized dataset
standardized_data.to_csv(
    "zscore_standardized_data.csv",
    index=False
)
```

# Q8. Calculate Gini coefficient
```
# Function to calculate Gini coefficient
def gini_coefficient(values):

    # Convert values into NumPy array
    x = np.array(values, dtype=float)

    # Remove missing values
    x = x[~np.isnan(x)]

    # Gini requires non-negative values
    if np.any(x < 0):
        raise ValueError("Gini coefficient requires non-negative values")

    # Handle empty or zero-valued data
    if len(x) == 0:
        return np.nan

    if np.sum(x) == 0:
        return 0

    # Sort values in ascending order
    x = np.sort(x)

    n = len(x)

    # Calculate Gini coefficient
    index = np.arange(1, n + 1)

    gini = (
        2 * np.sum(index * x) / (n * np.sum(x))
    ) - ((n + 1) / n)

    return gini


# Calculate Gini for all four pollutants
print("GINI COEFFICIENT OF AIR POLLUTANTS")

for col in columns:

    gini = gini_coefficient(clean_data[col])

    print(col, ":", round(gini, 4))


# Save numerical summary
summary.to_csv("numerical_summary.csv")

# Save measures of spread
spread.to_csv("measures_of_spread.csv")

print("All analysis results saved successfully!")
```



**Output**
# Loading the dataset and  preprocessing
<img width="718" height="608" alt="image" src="https://github.com/user-attachments/assets/8d6cbaed-a91f-4562-9fa1-0bf37ca8648d" />

# Distribution analysis using histograms
<img width="977" height="611" alt="image" src="https://github.com/user-attachments/assets/ca3e6ee4-cba0-4b62-af50-ec7801a82628" />

# Numerical summary
<img width="712" height="110" alt="image" src="https://github.com/user-attachments/assets/54eb4ae1-8267-4b3e-95d5-74a0c8efdb56" />

# Measures of spread
<img width="620" height="208" alt="image" src="https://github.com/user-attachments/assets/b6b40a75-5969-4b0b-a72c-bc1208ab8f51" />

# Construct box plots and detect outliers using IQR
<img width="1087" height="591" alt="image" src="https://github.com/user-attachments/assets/9b632a95-44db-41b1-b57e-81eef038decf" />

# Outlier detection using IQR
<img width="821" height="583" alt="image" src="https://github.com/user-attachments/assets/cd0405b7-4365-47bb-acb8-0a0deeb4ca3e" />

# Min-Max scaling
<img width="593" height="262" alt="image" src="https://github.com/user-attachments/assets/896706a4-b352-4e49-bb9d-c8e654273b49" />

# Z-score standardization using StandardScaler
<img width="610" height="387" alt="image" src="https://github.com/user-attachments/assets/2665f318-0cc4-4d51-aa85-7e4e81b6828f" />

# Gini coefficient
<img width="413" height="87" alt="image" src="https://github.com/user-attachments/assets/f967f244-194f-48af-85f6-36bacd27a3b7" />



**Interpretation**

1) PM2.5 and NO₂ show a strong positive correlation, suggesting that both pollutants increase together, likely due to vehicle and industrial emissions.

2) Ozone levels are comparatively lower than PM2.5 and PM10

3) PM2.5 and PM10 pollutants show considerable variation in their recorded concentrations throughout the observation period.

**Result**

The dataset was successfully loaded and processed to extract pollutant-wise and station-wise air quality data for Delhi.


