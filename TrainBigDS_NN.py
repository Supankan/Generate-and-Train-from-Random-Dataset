import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
from sklearn.metrics import r2_score

# Load the dataset
file_path = os.path.join('', 'randGenBigDS1.csv')
data = pd.read_csv(file_path)

# Split the dataset into features (X) and target variable (Y)
X = data.drop('Y', axis=1)
Y = data['Y']

# Split the dataset into training (90%) and testing (10%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))  # Assuming it's a regression task

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the neural network model
model.fit(X_train_scaled, Y_train, epochs=300, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model on test data and calculate R-squared
loss = model.evaluate(X_test_scaled, Y_test)
predictions = model.predict(X_test_scaled)
r_squared = r2_score(Y_test, predictions)
print(f"Model Loss: {loss}")
print(f"R-squared: {r_squared}")

# Save the trained model as an .h5 file
model_dir = os.path.join('', 'BigDSmodel_NN.keras')
model.save(model_dir)

print("Model saved as 'BigDSmodel_nn.keras' in the current directory.")
