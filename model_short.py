import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox, normal_ad
import logging
import sys
import os
from pymongo import MongoClient

logging.getLogger().setLevel(logging.INFO)

# --- Load data from MongoDB ---
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db_name = os.getenv("MONGO_DB", "weather_data_iso")
db = client[db_name]

args = sys.argv[1:]
for arg in args:
    arg.encode('utf-8').decode('utf-8')

date_start = sys.argv[1]
date_end = sys.argv[2]
city_name = sys.argv[3]

cursor = db.days.find({
    "city": city_name,
    "datetime": {
        "$gte": date_start,
        "$lte": date_end
    }
})

# --- Prepare the temperature series ---
data = list(cursor)
df = pd.DataFrame(data)
temp_series = df['temp']  # assuming 'temp' is the temperature column
temp_series = temp_series.dropna().squeeze()

# --- Check for stationarity using the ADF test ---
result = adfuller(temp_series)
adf_statistic, p_value = result[0], result[1]
adf_message = f"ADF Statistic: {adf_statistic} | p-value: {p_value}"
logging.info(adf_message)

temp_series_diff = temp_series

# --- Define and fit the AutoReg (AR) model ---
lags = 1  # Choose the number of lags (AR(1) in this case, can be adjusted)
model = AutoReg(temp_series_diff, lags=lags)
ar_model_results = model.fit()

# --- Check the residuals ---
residuals = ar_model_results.resid

# Test if residuals are independent (Ljung-Box test)
ljung_box_test = acorr_ljungbox(residuals, lags=[lags])
ljung_box_message = f"Ljung-Box Test p-value: {float(ljung_box_test['lb_pvalue'])}"
logging.info(ljung_box_message)

# Test if residuals follow a normal distribution (using statsmodels' normal_ad test)
normal_ad_test = normal_ad(residuals)
normal_ad_message = f"Normality Test (Anderson-Darling) p-value: {normal_ad_test}"
logging.info(normal_ad_message)

# --- Forecast the next value ---
forecast = ar_model_results.predict(start=len(temp_series_diff), end=len(temp_series_diff) + 6)

# --- Save results to a text file ---
file_path = "path/to/your/file"  # Replace with your desired file path
mae_msg = f"Mean Absolute Error: Some_value_here"  # You can update this with the actual MAE value

# Convert predictions to Fahrenheit and Celsius
pred_temp_fahrenheit = [round(temp, 2) for temp in forecast]
pred_temp_celsius = [round((temp - 32) * 5 / 9, 2) for temp in pred_temp_fahrenheit]

# Generate forecast dates
forecast_dates = [pd.to_datetime(date_end) + pd.Timedelta(days=i) for i in range(1, 8)]
forecast_df = pd.DataFrame({'date': forecast_dates, 'temp_celsius': pred_temp_celsius, 'temp_fahrenheit': pred_temp_fahrenheit})

logging.info("Prediction finished.")

# --- Log final results ---
final_msg = f"""
Predicted temperature for {city_name} from {date_start} to {date_end}:
{pred_temp_fahrenheit}°F  ({pred_temp_celsius}°C).
"""
logging.info(final_msg)

# --- Save results to files ---
log_filename = f"{city_name}-{date_start}-{date_end}"
log_folder = "/app/logs"
file_path = os.path.join(log_folder, log_filename)

forecast_df.to_csv(f"{file_path}-results.csv", index=False)

# Write all results into the file
with open(f"{file_path}-measurements.txt", "w") as file:
    file.write(mae_msg + "\n")  # Write MAE message
    file.write(adf_message + "\n")  # Write ADF test result
    file.write(ljung_box_message + "\n")  # Write Ljung-Box test result
    file.write(normal_ad_message + "\n")  # Write Shapiro-Wilk test result

logging.info("Results files generated.")
