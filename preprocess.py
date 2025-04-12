import pandas as pd
import numpy as np

#Load the Excel file and data
file_path = "Ouse93-96 - Student.xlsx"
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name="1993-96")

#First column is date and rest are the stations/rivers and then drop first row
df.columns = ["Date"] + list(df.iloc[0, 1:])
df = df.iloc[1:].reset_index(drop=True)

#Separate the Date column from the rest of the columns
date_column = df["Date"]
data_columns = df.iloc[:, 1:]

# Convert all values to numeric or NaN
data_columns = data_columns.apply(pd.to_numeric, errors='coerce')
# Handle missing values by interpolating
data_columns.interpolate(method='linear', inplace=True)

#Calculate IQR and outliers
Q1 = data_columns.quantile(0.25)
Q3 = data_columns.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

#Identify outliers
outlier_mask = (data_columns < lower_bound) | (data_columns > upper_bound)
outlier_counts = outlier_mask.sum(axis=1)
#Remove a reasonable number of extreme outliers
num_outliers_to_remove = int(len(data_columns) * 0.03) #3% of the data
outlier_indices = outlier_counts.nlargest(num_outliers_to_remove).index
data_cleaned = data_columns.drop(index=outlier_indices).reset_index(drop=True) #Remove outlier rows
date_cleaned = date_column.drop(index=outlier_indices).reset_index(drop=True)

#Min Max scale to [0.1, 0.9]
data_standardised = data_cleaned.copy()
for col in data_standardised.columns:
    min_val = data_standardised[col].min()
    max_val = data_standardised[col].max() #Calculate min and max for each column
    data_standardised[col] = 0.8 * (data_standardised[col] - min_val) / (max_val - min_val) + 0.1
#Concatenate the Date column back with the standardised data
df_standardised = pd.concat([date_cleaned, data_standardised], axis=1)

#Split data into Training (50%), Validation (25%), and Testing (25%)
train_size = int(0.5 * len(df_standardised))
val_size = int(0.25 * len(df_standardised))
#Create the dataframes for training, validation, and testing
df_train = df_standardised.iloc[:train_size]
df_val = df_standardised.iloc[train_size:train_size + val_size]
df_test = df_standardised.iloc[train_size + val_size:]

#Save preprocessed data into one Excel file with multiple sheets
output_file = "preprocessed_data.xlsx"
with pd.ExcelWriter(output_file) as writer:
    df_train.to_excel(writer, sheet_name="Train", index=False)
    df_val.to_excel(writer, sheet_name="Validation", index=False)
    df_test.to_excel(writer, sheet_name="Test", index=False)

print(f"Preprocessing complete! Data saved in {output_file} with separate sheets for Train, Validation, and Test.")