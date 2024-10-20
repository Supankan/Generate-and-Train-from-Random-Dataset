import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Load the dataset
file_path = os.path.join('', 'randGenBigDS1.csv')
data = pd.read_csv(file_path)

# Split the dataset into features (X) and target variable (Y)
X = data.drop('Y', axis=1)
Y = data['Y']

# Split the dataset into training (80%) and testing (10%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust hyperparameters here

# Train the Random Forest model
model.fit(X_train, Y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(Y_test, predictions)
r_squared = r2_score(Y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r_squared}")

# You can save the model if needed
model_dir = os.path.join('', 'BigDSmodel_RF.pkl')
with open(model_dir, 'wb') as file:
    pickle.dump(model, file)

print("Model trained using Random Forest and metrics calculated.")
