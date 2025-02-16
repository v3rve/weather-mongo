import pandas as pd
import logging
import sys
import os
import numpy as np
from xgboost import XGBRegressor
from pymongo import MongoClient
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, explained_variance_score, max_error
from sklearn.preprocessing import StandardScaler

logging.getLogger().setLevel(logging.INFO)

# --- Load data from MongoDB ---
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db_name = os.getenv("MONGO_DB", "weather_data_iso")
db = client[db_name]

# --- Read script arguments ---
date_start, date_end, city_name = sys.argv[1:4]

# --- Retrieve data from MongoDB ---
cursor = db.days.find({
    "city": city_name,
    "datetime": {"$gte": date_start, "$lte": date_end}
})

data = list(cursor)
if not data:
    logging.error(f"No data found for {city_name} between {date_start} and {date_end}.")
    sys.exit(1)

df = pd.DataFrame(data)

# --- Feature engineering ---
df['datetime'] = pd.to_datetime(df['datetime'])
df['month'] = df['datetime'].dt.month
df['day_of_year'] = df['datetime'].dt.dayofyear
df['day_of_week'] = df['datetime'].dt.weekday

# Select numeric features dynamically
numeric_features = df.select_dtypes(include=['number']).columns.tolist()
excluded_features = ["tempmax", "tempmin", "feelslikemax", "feelslikemin", "feelslike", "datetime"]
numeric_features = [feature for feature in numeric_features if feature not in excluded_features]

# Define the target variable
target = 'temp'

# Generate lagged features for forecasting
lag_range = range(1, 4)
for feature in numeric_features:
    for lag in lag_range:
        df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

df = df.fillna(0)  # Replace NaN with 0

# --- Prepare data for training ---
X = df[[f'{feature}_lag{lag}' for feature in numeric_features if feature != target for lag in lag_range]]
y = df[target]

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logging.info("Data preparation completed.")

# --- Define hyperparameter space ---
param_dist = {
    'n_estimators': np.arange(50, 300, 50),
    'learning_rate': np.linspace(0.01, 0.2, 5),
    'max_depth': np.arange(3, 10, 2),
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# --- Train model with RandomizedSearch ---
xgb_model = XGBRegressor(objective='reg:squarederror')

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

logging.info(f"Best parameters: {random_search.best_params_}")

# --- Make predictions ---
y_pred = best_model.predict(X_test)

# Convert predictions to Fahrenheit and Celsius
y_pred_fahrenheit = [round(temp, 2) for temp in y_pred]
y_pred_celsius = [round((temp - 32) * 5 / 9, 2) for temp in y_pred]

# --- Compute evaluation metrics ---
mae = round(float(mean_absolute_error(y_test, y_pred)), 8)
evs = round(float(explained_variance_score(y_test, y_pred)), 8)
max_err_value = round(float(max_error(y_test, y_pred)), 8)

logging.info(f"MAE: {mae}")
logging.info(f"Explained Variance Score: {evs}")
logging.info(f"Max Error: {max_err_value}")

# --- Generate forecast dates ---
nr_days_predict = len(y_pred) + 1
forecast_dates = [pd.to_datetime(date_end) + pd.Timedelta(days=i) for i in range(1, nr_days_predict)]
forecast_df = pd.DataFrame({'date': forecast_dates, 'temp_celsius': y_pred_celsius, 'temp_fahrenheit': y_pred_fahrenheit})

# --- Log final results ---
final_msg = f"""
Predicted temperature for {city_name} based on data from {date_start} to {date_end}:
{y_pred_fahrenheit}°F  ({y_pred_celsius}°C).
"""
logging.info(final_msg)

# --- Save results to files ---
log_filename = f"{city_name}-{date_start}-{date_end}"
log_folder = "/app/logs"
file_path = os.path.join(log_folder, log_filename)

forecast_df.to_csv(f"{file_path}-results.csv", index=False)

with open(f"{file_path}-measurements.txt", "w") as file:
    file.write(f"MAE: {mae}\n")
    file.write(f"Explained Variance Score: {evs}\n")
    file.write(f"Max Error: {max_err_value}\n")

logging.info("Results file generated.")
