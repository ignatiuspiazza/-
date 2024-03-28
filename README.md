# -
スマート工場は、エッジコンピューティングと機械学習を活用して、製造工程のリアルタイムモニタリングと最適化を行い、生産性の向上と不良品の削減を実現します。
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import random

# Simulated sensor data generation
def generate_sensor_data(num_samples=1000):
    """Generate simulated sensor data for temperature, pressure, and vibration."""
    temperatures = np.random.normal(100, 10, num_samples)
    pressures = np.random.normal(10, 2, num_samples)
    vibrations = np.random.normal(0.5, 0.1, num_samples)
    # Simulate defects
    defects = np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])
    data = pd.DataFrame({
        'temperature': temperatures,
        'pressure': pressures,
        'vibration': vibrations,
        'defect': defects
    })
    return data

# Load and preprocess data
data = generate_sensor_data()
features = data.drop('defect', axis=1)
labels = data['defect']

# Split the dataset into training and testing sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Machine Learning model for predicting defects
def train_model(features_train, labels_train):
    """Train a machine learning model to predict defects."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features_train, labels_train)
    return model

# Train the model
model = train_model(features_train, labels_train)

# Evaluate the model
predictions = model.predict(features_test)
print(f"Model accuracy: {accuracy_score(labels_test, predictions)}")

# Real-time monitoring and prediction
def real_time_monitoring(model, new_data):
    """Simulate real-time monitoring and prediction of defects."""
    prediction = model.predict([new_data])
    if prediction[0] == 1:
        print("Warning: Potential defect detected!")
    else:
        print("Status: Normal")

# Simulate new sensor data
new_sensor_data = [100 + random.gauss(0, 10), 10 + random.gauss(0, 2), 0.5 + random.gauss(0, 0.1)]
real_time_monitoring(model, new_sensor_data)

# This is a simplified demo. In a real-world scenario, you'd deploy the model to an edge computing device
# where it can make predictions in real-time based on the data from the factory's sensors.
