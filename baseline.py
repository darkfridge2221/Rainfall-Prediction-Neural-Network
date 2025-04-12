import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load the preprocessed dataset
file_path = "C:/Users/User/Downloads/preprocessed_data.xlsx"
df = pd.read_excel(file_path, sheet_name=None)

#Extract training, validation, and test sets
train_data = df['Train']
val_data = df['Validation']
test_data = df['Test']

#Merge train and validation for regression training (as suggested in slides)
train_val_data = pd.concat([train_data, val_data])

#Define predictors and predictand
predictors = ["Crakehill", "Skip Bridge", "Westwick"]
predictand = "Skelton"

#Prepare training and test data
X_train = train_val_data[predictors]
y_train = train_val_data[predictand]
X_test = test_data[predictors]
y_test = test_data[predictand]

#Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

#Evaluate the model
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
msre = np.mean(((y_test - y_pred_test) / y_test) ** 2)
ce = 1 - (np.sum((y_test - y_pred_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

#Print model performance
print(f"Test RMSE: {test_rmse:.6f}")

#Plot time series
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, y_test, label="Actual Skelton Flow", color='blue')
plt.plot(test_data.index, y_pred_test, label="Predicted Skelton Flow", color='red')
plt.xlabel("Date")
plt.ylabel("Skelton Flow")
plt.title("Time Series: Actual vs Predicted Skelton Flow")
plt.legend()
plt.show()

#Plot scatter graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.5, label="Predicted vs Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Fit")
plt.xlabel("Actual Skelton Flow")
plt.ylabel("Predicted Skelton Flow")
plt.title("Linear Regression: Actual vs Predicted Skelton Flow")
plt.legend()
plt.show()