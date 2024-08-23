
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Save the 'Id' column from the test dataset
test_ids = test['Id']

# Check the first few rows and column names
print(train.head())
print("Columns in the training dataset:", train.columns)

# Handle missing values
missing_values = train.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values)
plt.xticks(rotation=90)
plt.title('Missing Values')
plt.show()

for col in missing_values.index:
    if train[col].dtype == 'object':
        train[col].fillna(train[col].mode()[0], inplace=True)
        test[col].fillna(test[col].mode()[0], inplace=True)
    else:
        train[col].fillna(train[col].median(), inplace=True)
        test[col].fillna(test[col].median(), inplace=True)

# Encode categorical variables
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Align train and test datasets by the columns
train, test = train.align(test, join='left', axis=1)

# Ensure SalePrice is not in the test DataFrame
if 'SalePrice' in test.columns:
    test.drop(columns=['SalePrice'], inplace=True)

# Fill any remaining missing values in test
test.fillna(0, inplace=True)

# Ensure SalePrice is still in the train DataFrame
if 'SalePrice' not in train.columns:
    raise KeyError("'SalePrice' column is missing from the training data.")
else:
    print("'SalePrice' column is present.")

# Split the data
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
test_scaled = scaler.transform(test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Save the trained model as a .h5 file
model_filename = 'random_forest_model.h5'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# Make predictions on test data
test_predictions = model.predict(test_scaled)

# Prepare submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})
#submission.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
