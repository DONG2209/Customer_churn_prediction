# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and Prepare the Data
# Load the dataset into pandas dataframes
data_20 = pd.read_csv("Churn-bigml-20.csv")
data_80 = pd.read_csv("Churn-bigml-80.csv")

# Combine the two datasets if needed
data = pd.concat([data_20, data_80])

# Data Preprocessing
# Encoding categorical variables (International plan, Voice mail plan, and Churn)
label_encoder = LabelEncoder()
data['State'] = label_encoder.fit_transform(data['State'])
data['International plan'] = label_encoder.fit_transform(data['International plan'])
data['Voice mail plan'] = label_encoder.fit_transform(data['Voice mail plan'])
data['Churn'] = label_encoder.fit_transform(data['Churn'])

# Step 2: Exploratory Data Analysis (EDA)
# Perform EDA to understand the data
# Examine distributions, correlations, and visualize data

# Step 3: Data Splitting
# Split the data into training and testing sets
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and Train the Model
# Use a machine learning model (Random Forest Classifier in this example)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print(report)
