import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import csv

# Load real data if available, otherwise simulate
try:
    X_train = []
    y_train = []
    with open('cv_data.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            X_train.append([
                float(row['education_level']),
                float(row['years_of_experience']),
                float(row['skills_count']),
                float(row['certifications_count'])
            ])
            y_train.append(float(row['score']))
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print("Loaded real data from cv_data.csv")
except FileNotFoundError:
    # Simulated realistic data
    X_train = np.array([
        [2, 5, 6, 2],  # Strong CV: degree, 5 years, 6 skills, 2 certs
        [1, 2, 3, 1],  # Average: some edu, 2 years, 3 skills, 1 cert
        [0, 1, 2, 0],  # Weak: no edu, 1 year, 2 skills, no certs
        [2, 10, 8, 3], # Excellent: degree, 10 years, 8 skills, 3 certs
        [1, 3, 4, 1],  # Moderate: some edu, 3 years, 4 skills, 1 cert
        [0, 0, 1, 0],  # Minimal: no edu, no exp, 1 skill, no certs
        [2, 7, 5, 2],  # Good: degree, 7 years, 5 skills, 2 certs
        [1, 4, 7, 1]   # Above average: some edu, 4 years, 7 skills, 1 cert
    ])
    y_train = np.array([85, 60, 40, 95, 65, 30, 80, 70])  # Realistic scores
    print("Using simulated data")

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('cv_score_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Test with a sample
sample = np.array([[2, 5, 6, 2]])  # Example: degree, 5 years, 6 skills, 2 certs
predicted_score = model.predict(sample)[0]
print(f"Predicted score for sample CV: {predicted_score:.2f}")

print("Model trained and saved as cv_score_model.pkl!")