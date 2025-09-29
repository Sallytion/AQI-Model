# 🇮🇳 India AQI Dashboard & Prediction System

A real-time Air Quality Index (AQI) monitoring and prediction dashboard for Indian cities, powered by machine learning and live government data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-here.streamlit.app)

## 🌟 Live Demo Features

- **📊 Real-Time Dashboard**: Live AQI data from 3000+ monitoring stations
- **🤖 Smart Predictions**: ML-powered AQI forecasting with weather integration  
- **🌍 Interactive Maps**: Explore pollution levels across Indian cities
- **📈 Historical Analysis**: Trends from 2015-2024 datasets
- **🏥 Health Impact**: Personalized health recommendations

## 🚀 Quick Start

### Option 1: Run Locally
```bash
# Clone and run
git clone https://github.com/Sallytion/AQI-Model.git
cd AQI-Model
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Deploy on Streamlit Cloud
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with: `main` branch, `app.py` file

## 🎯 Key Features

### Smart Prediction Tool
- **Real-time data**: Government API integration
- **Weather-enhanced**: Historical/current/forecast weather data
- **Individual pollutants**: PM2.5, PM10, NO2, SO2, O3, CO predictions
- **City-specific**: Accurate coordinates for 45+ Indian cities
- **High accuracy**: R² = 0.951, trained on 6,236+ real measurements

### Dashboard Analytics  
- **Multi-city analysis**: Delhi NCR trends (2022-2024)
- **State-wise insights**: 2021 comprehensive data
- **Real-time monitoring**: Live API data with filtering
- **Interactive visualizations**: Plotly charts and maps
## 📊 Tech Stack

- **Frontend**: Streamlit (Python web framework)
- **ML**: Scikit-learn (Random Forest models)  
- **Data**: Pandas, NumPy
- **Visualization**: Plotly (interactive charts & maps)
- **APIs**: Government of India CPCB, Open Meteo Weather
- **Deployment**: Streamlit Community Cloud + Git LFS

## 📁 Project Structure

```
AQI-Model/
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data processing utilities  
├── retrain_model.py       # Model training script
├── requirements.txt       # Python dependencies
├── aqi_weather_model.joblib    # Pre-trained ML model (Git LFS)
├── aqi_weather_features.joblib # Feature encodings
├── aqi_weather_targets.joblib  # Target encodings
├── aqi_weather_metrics.joblib  # Model metrics
├── *.csv                  # Historical datasets
└── 2015-2020_data/       # Training data
    └── city_day.csv
```

## ⚙️ Model Performance

- **Accuracy**: R² = 0.951 (95.1% variance explained)
- **Error Rate**: MAE = 13.57 AQI points  
- **Training Data**: 6,236 validated measurements
- **Features**: Weather data + Pollutant concentrations
- **Validation**: Cross-validated on real monitoring station data

## 🏥 AQI Categories (India CPCB Standard)

| Category | AQI Range | Health Impact | Color |
|----------|-----------|---------------|-------|
| Good | 0-50 | Minimal Impact | 🟢 |
| Satisfactory | 51-100 | Minor breathing discomfort | 🟡 |  
| Moderate | 101-200 | Breathing discomfort | 🟠 |
| Poor | 201-300 | Respiratory illness | 🔴 |
| Very Poor | 301-400 | Respiratory effects | 🟣 |
| Severe | 401+ | Emergency conditions | ⚫ |

## 🔗 Data Sources

- **Live API**: [Government of India CPCB](https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69)
- **Weather**: [Open Meteo API](https://open-meteo.com)
- **Historical**: Official government datasets (2015-2024)

---

💡 **Perfect for**: Environmental research, public health monitoring, air quality awareness, and educational demonstrations of ML in environmental science.