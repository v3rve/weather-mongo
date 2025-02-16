import requests
import logging
import json
import os
import sys
import pymongo
import pandas as pd
import subprocess
from datetime import datetime, timedelta
from functions.custom_functions_app import *

logging.basicConfig(level=logging.INFO)

# --- Load Configuration ---
config_data = json.load(open('config/config_cred.json'))
api_key = config_data["api_key"]
base_url = config_data["base_url"]

coordinates_data = json.load(open('config/config_locations.json', encoding='utf-8'))
list_cities = list(coordinates_data.keys())
list_cities_model = list(coordinates_data.keys())

# --- Get Environment Variables ---
date_start = date_validation(os.getenv("DATE_START", "2024-04-11"))
date_end = date_validation(os.getenv("DATE_END", "2024-04-18"))
date_length_check(date_start, date_end)

client_address = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
db_name = os.getenv("MONGO_DB", "weather_data_iso")
full_refresh_param = int(os.getenv("FULL_REFRESH", 0))
choice_model_param = int(os.getenv("CHOICE_MODEL", 0))

# --- Connect to MongoDB ---
db_weather_data = connect_to_db(client_address=client_address, db_name=db_name)

# --- Generate Date Range ---
date_list = pd.date_range(start=date_start, end=date_end).to_list()
time_range_diff = (date_list[-1] - date_list[0]).days

# --- Check Database Before Fetching New Data ---
if full_refresh_param == 1:
    for city_name in list_cities.copy():  # Copy to allow modification inside loop
        city_passed = True

        existing_dates = [
            doc["datetime"] for doc in db_weather_data.days.find(
                {"city": city_name, "datetime": {"$gte": date_start, "$lte": date_end}},
                {"datetime": 1, "_id": 0}
            )
        ]

        if not existing_dates:
            logging.info(f"No data found for {city_name}. Fetching new data.")
            continue

        existing_dates = [datetime.strptime(date_str, "%Y-%m-%d") for date_str in existing_dates]
        min_date, max_date = min(existing_dates), max(existing_dates)

        if (max_date - min_date).days != time_range_diff:
            logging.info(f"Incomplete data range for {city_name}. Adjusting request range.")
            if datetime.strptime(date_start, "%Y-%m-%d") <= max_date < datetime.strptime(date_end, "%Y-%m-%d"):
                date_start = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")
            elif datetime.strptime(date_end, "%Y-%m-%d") >= min_date > datetime.strptime(date_start, "%Y-%m-%d"):
                date_end = (min_date - timedelta(days=1)).strftime("%Y-%m-%d")

            city_passed = False

        if city_passed:
            list_cities.remove(city_name)
            logging.info(f"Skipping {city_name}: All data already exists.")

    logging.info(f"Fetching data for: {list_cities}")

    # --- Fetch New Data from API ---
    for city_name in list_cities:
        lat, lon = coordinates_data[city_name]['latitude'], coordinates_data[city_name]['longitude']
        api_url = f'{base_url}{lat},{lon}/{date_start}/{date_end}?key={api_key}'

        logging.info(f"Fetching data for {city_name} from {api_url}")
        response = requests.get(api_url)

        if not check_response(response.text):
            logging.warning(f"API request failed for {city_name}. Skipping.")
            continue

        weather_data = response.json()
        days_db = weather_data.get('days', [])

        for day in days_db:
            hours = day.get("hours", [])
            day.pop('hours', None)  # Remove hours from the day record before inserting
            dt = day["datetime"]
            day.update({"city": city_name, "datetime": dt})

            if not db_weather_data.days.find_one({"city": city_name, "datetime": dt}):
                db_weather_data.days.insert_one(day)
                logging.info(f"Inserted day data: {dt} - {city_name}")
            # --- Insert Hourly Data ---
            for hour in hours:
                ht_hour = hour["datetime"]
                hour.update({"city": city_name, "date": dt, "datetime": ht_hour})

                existing_hour_doc = db_weather_data.hours.find_one({
                    "city": city_name,
                    "date": dt,
                    "datetime": ht_hour
                })

                if existing_hour_doc:
                    logging.info(f"Hourly data already exists for {ht_hour} - {city_name}")
                else:
                    db_weather_data.hours.insert_one(hour)
                    logging.info(f"Inserted hourly data: {ht_hour} - {city_name}")

# --- Run Forecasting Model ---
for city_name in list_cities_model:
    model_args = [date_start, date_end, city_name]

    cursor = db_weather_data.days.find({"city": city_name, "datetime": {"$gte": date_start, "$lte": date_end}})
    data = list(cursor)

    if not data:
        logging.warning(f"No data available for {city_name}. Skipping model execution.")
        continue

    df = pd.DataFrame(data)

    if choice_model_param == 1:
        logging.info(f"Running Gradient Boosting model for {city_name}")
        subprocess.run(["python", "model.py"] + model_args)
    else:
        logging.info(f"Running SARIMA model for {city_name}")
        subprocess.run(["python", "model_short.py"] + model_args)
