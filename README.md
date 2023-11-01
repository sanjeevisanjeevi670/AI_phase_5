# AI_phase_5
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('https://tn.data.gov.in/resource/company-master-data-tamil-nadu-upto-28th-february-2019')

# Preprocess the data
df = df.dropna()
df['Company Registration Date'] = pd.to_datetime(df['Company Registration Date'])
df['Year'] = df['Company Registration Date'].dt.year

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['Year']], df['Company Registrations'], test_size=0.25, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
r2_score = r2_score(y_test, y_pred)
print('R-squared score:', r2_score)

# Create a function to predict the number of company registrations for a given year
def predict_company_registrations(year):
  """
  Predicts the number of company registrations for a given year.

  Args:
    year: The year for which to predict the number of company registrations.

  Returns:
    The predicted number of company registrations.
  """

  X_new = np.array([[year]])
  y_pred = model.predict(X_new)
  return y_pred[0][0]

# Predict the number of company registrations for the year 2024
predicted_company_registrations_2024 = predict_company_registrations(2024)
print('Predicted number of company registrations for 2024:', predicted_company_registrations_2024)
