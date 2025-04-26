import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Training data
X_train_full = np.array([
    [0, 0, 0, 0], [1, 2, 3, 0], [2, 5, 4, 1], [3, 7, 5, 2], [4, 10, 6, 3],
    [3, 8, 4, 1], [4, 12, 5, 2], [5, 15, 7, 4], [1, 3, 2, 0], [2, 6, 4, 1],
    [5, 9, 5, 2], [3, 4, 3, 1], [4, 18, 6, 3], [1, 1, 2, 0], [2, 5, 4, 1],
    [5, 13, 6, 3], [3, 6, 5, 1], [4, 11, 5, 2], [5, 8, 5, 2], [1, 3, 3, 0],
    [4, 14, 6, 3], [5, 20, 10, 5], [3, 9, 7, 3]
])
y_train_full = np.array([0, 15, 30, 45, 60, 40, 60, 80, 15, 35, 60, 30, 75, 10, 30, 70, 40, 55, 55, 20, 70, 100, 75])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Feature engineering
numerical_features = [1, 2, 3]  # years_experience, num_skills, num_certifications
categorical_features = [0]  # education_level

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(categories=[[0, 1, 2, 3, 4, 5]], handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# Define model
gbr = GradientBoostingRegressor(random_state=42, loss='squared_error')

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('gbr', gbr)
])

# Hyperparameter grid
param_grid = {
    'gbr__n_estimators': [50, 100],
    'gbr__learning_rate': [0.01, 0.1],
    'gbr__max_depth': [2, 3],
    'gbr__min_samples_split': [2, 3]
}

# Perform GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_
print(f"Best hyperparameters: {grid_search.best_params_}")

# Evaluate with cross-validation (3-fold due to small dataset)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=3, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Average CV R² score: {np.mean(cv_scores):.2f} (±{np.std(cv_scores):.2f})")

# Evaluate on the test set
y_pred = best_model.predict(X_test)
y_pred = np.clip(y_pred, 0, 100)  # Ensure predictions are in range 0–100
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test set MSE: {mse:.2f}")
print(f"Test set MAE: {mae:.2f}")
print(f"Test set R² score: {r2:.2f}")

# Analyze prediction distribution
print(f"Prediction distribution: min={np.min(y_pred):.2f}, max={np.max(y_pred):.2f}, mean={np.mean(y_pred):.2f}")

# Save the model and preprocessor
with open('scaler.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)
with open('cv_score_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Improved model and preprocessor saved as cv_score_model.pkl and scaler.pkl")