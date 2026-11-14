from sklearn.datasets import fetch_california_housing

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import joblib

# STEP 1: Load Dataset
print("Loading California Housing Dataset...")

data = fetch_california_housing()
# Convert dataset into DataFrame
df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

# Add target column
df["Price"] = data.target

print("Dataset Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Description:")
print(df.describe())


# STEP 2: Separate Features and Target
X = df.drop("Price", axis=1)
y = df["Price"]

# STEP 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# training a mode
model=RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# STEP 5: Train Model
print("\nTraining Model...")

model.fit(X_train, y_train)

print("Model Training Completed!")

# STEP 6: Prediction
y_pred = model.predict(X_test)

# STEP 7: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print(f"Average error${mae *100000:,.00f}")


# Save Model
joblib.dump(model,"house_model.joblib")
joblib.dump(list(X.columns),"house_feature.joblib")


