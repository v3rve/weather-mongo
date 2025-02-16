# 🌤️ weather-mongo  

**Short-term temperature forecasting using MongoDB and Machine Learning**  

## 📜 Table of Contents  
- [📌 Project Overview](#project-overview)  
- [🛠️ Tech Stack](#tech-stack)  
- [⚙️ Installation & Setup](#installation--setup)  
- [📊 Example Forecast Output](#example-forecast-output)  
- [📂 Suggested Project Structure](#suggested-project-structure)  
- [📄 License](#license)  

## Project Overview
This project retrieves weather data from an API, stores it in **MongoDB**, and applies **Autoregressive model** & **Gradient Boosting** to predict short-term temperature changes.  

## Tech Stack
- **MongoDB** – NoSQL database for storing weather data  
- **Python** – Main programming language  
- **AR & Gradient Boosting** – Forecasting models for temperature prediction  
- **Docker & Docker Compose** – Containerization for easy deployment  

## Installation & Setup

### 1️⃣ Clone the repository  
```sh
git clone https://github.com/yourusername/weather-mongo.git
cd weather-mongo
```

### 2️⃣ Install dependencies  
```sh
pip install -r requirements.txt
```
### 3️⃣ Retrieve your personal API keys 
1. Please register here to get API key:
https://www.visualcrossing.com/weather-api/
2. Update the file config_cred.json with the new API key.

### 4️⃣ Set up environment variables & API keys  
Edit the **config/config_cred.json** file with your API keys.  
Update the following variables in the docker-compose.yml file:

- MONGO_URI: MongoDB connection string
- MONGO_DB: Database name
- DATE_START & DATE_END: Date range for data retrieval
- FULL_REFRESH: Full refresh param
- CHOICE_MODEL: Set to 1 to use a more complex Gradient Boosting Model (better for longer-term forecasts). Otherwise, a simpler AR model will be used.
- RESULT_FILE_PATH: Path for saving results

- Please update REPO_FOLDER_PATH with the local folder, where the results should be exported

### 5️⃣ Run the application  


#### 🔹 Run with Docker  
```sh
docker-compose up --build
```

---

## Example Forecast Output
![image](https://github.com/user-attachments/assets/7e03366a-74bb-4d83-afbd-941cd72dd2cc)


---

## Suggested Project Structure

```bash
weather-mongo/
│── config/                    # Configuration files
│   ├── config_cred.json       # Credentials (API keys, MongoDB access)
│   ├── config_locations.json  # Location settings for weather data
│── functions/                 # Custom functions for processing & modeling
│   ├── custom_functions_app.py # Helper functions for the main app
│── models/                    # Machine learning models
│   ├── model.py               # SARIMA & boosting models
│   ├── model_short.py         # Short-term forecasting models
│── docker/                    # Docker-related files
│   ├── Dockerfile             # Docker image configuration
│   ├── docker-compose.yaml    # Docker Compose setup
│── app_main.py                # Main application entry point
│── requirements.txt           # Python dependencies
│── README.md                  # Documentation
```

## License
Distributed under the **MIT License**. See **LICENSE.txt** for more information.  
