# 🌦️ Weather Trend Forecasting

A data science and machine learning project to forecast weather trends using the **Global Weather Repository** dataset. The project includes data cleaning, exploratory data analysis (EDA), and the development of forecasting models such as **LSTM** and **Random Forest**.

## 📌 Features

✅ **Weather Forecasting** – Predicts temperature and precipitation trends based on historical weather data.  
✅ **Machine Learning Models** – Uses LSTM for time-series forecasting and Random Forest for comparative analysis.  
✅ **Geographical Analysis** – Analyzes temperature patterns across countries and continents.  
✅ **Air Quality Insights** – Examines the correlation between weather conditions and air quality indices.  
✅ **Climate Analysis** – Identifies long-term temperature trends that may indicate climate change.

## 🗺️ Dataset

The dataset used for this project is the **Global Weather Repository** available on Kaggle. It contains historical weather data, including temperature, humidity, precipitation, and air quality measures.

- **Global Weather Dataset**  
- **Features**: Temperature, Wind Speed, Precipitation, Humidity, Air Quality, etc.  
- **Source**: [Global Weather Repository on Kaggle](https://www.kaggle.com/datasets)

## 🛠️ Tech Stack

- **Python 3.11**
- **Pandas, NumPy** (for data manipulation and analysis)
- **Matplotlib, Seaborn** (for data visualization)
- **TensorFlow/Keras** (for model training - LSTM)
- **Scikit-learn** (for Random Forest and model evaluation)
- **Jupyter Notebook** (for exploratory analysis and reporting)

## 🚀 Installation & Setup

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/Abena-3565/Weather_Trend_Forecasting.git
cd Weather_Trend_Forecasting
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate  # For Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Jupyter Notebook
jupyter notebook
📊 Data Analysis & Model Development
Data Cleaning: Handling missing values, outliers, and normalizing numerical features.

Exploratory Data Analysis (EDA): Visualizing temperature trends, humidity patterns, and correlations between weather and air quality.

Forecasting Models: Using LSTM and Random Forest to predict weather trends.

LSTM for time-series forecasting.

Random Forest for comparison and model evaluation.

Advanced Analysis: Climate change detection, geographical weather patterns, and environmental impact of weather conditions.

🔍 Model Evaluation
Metrics: Models are evaluated using Root Mean Squared Error (RMSE) for accuracy.

Model Performance: Random Forest outperforms LSTM in terms of RMSE for this dataset.

🌍 Geographical & Climate Analysis
Geographical Patterns: Analyzed temperature distribution across different countries and continents.

Climate Change: Identified long-term trends in temperature that suggest possible climate change.

📡 Example Usage
To predict future weather trends using the trained model, you can run the following code:
import tensorflow as tf
import numpy as np
import pandas as pd

# Load the pre-trained model
model = tf.keras.models.load_model("weather_forecast_model.h5")

def predict_weather(input_data):
    data = np.array(input_data).reshape(1, -1)  # Adjust shape based on model input
    prediction = model.predict(data)
    return prediction

# Example input data for prediction
input_data = [25, 75, 1012, 5]  # Example values for temperature, humidity, pressure, and wind speed
print(predict_weather(input_data))
📷 Screenshots
Temperature Trend: A time-series plot showing the change in global temperature.

Weather Correlations: Heatmaps of correlations between temperature and other weather conditions like humidity and air quality.

🤝 Contributing
Contributions are welcome! Please follow these steps:
Fork the repo
Create a new branch (feature-new-idea)
Commit your changes
Submit a Pull Request

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

💡 Acknowledgments
Global Weather Repository
Kaggle
TensorFlow/Keras Community
OpenAI & Deep Learning Research

📩 Contact
For questions or suggestions, reach out:
📧 Email: abenezeralz659@gmail.com
🔗 LinkedIn: Abenezer Alemayehu LinkedIn
💻 GitHub: Abena-3565 Weather Trend Forecasting
📱 Phone: +251935651441
