# India AQI Dashboard

A comprehensive Streamlit dashboard for analyzing and predicting Air Quality Index (AQI) across India using multiple datasets.

## Features

### 📊 Dashboard (Analysis)
- Trend graphs from 2021–2024 datasets
- Good vs bad days analysis by city and state
- Real-time data overview with key metrics

### 🤖 Prediction Tool
- Machine learning models for AQI prediction
- Regression model for numeric AQI values
- Classification model for AQI categories
- Feature importance visualization

### 🌍 Real-Time Insights
- **Live API Integration**: Fetches real-time data from Government of India API
- **Automatic Fallback**: Uses local CSV if API is unavailable
- **Dynamic Filtering**: Filter by state and city with API-level filtering
- **Pollutant Analysis**: Distribution charts for different pollutants (PM10, PM2.5, NO2, SO2, CO, OZONE, NH3)
- **Station Mapping**: Interactive maps showing monitoring station locations
- **Data Refresh**: Manual refresh button for latest data
- **Summary Statistics**: Real-time metrics and counts

## Data Sources

The dashboard uses three data sources:

### Static Analysis Data:
1. **Year-wise Details of Air Quality Index (AQI) levels in DelhiNCR from 2022 to 2024.csv**
2. **StateUTs-wise Details of the Air Quality Index (AQI) of the Country during 2021.csv**

### Real-Time Data:
3. **Government of India API**: Live real-time AQI data from https://api.data.gov.in/
   - **Fallback**: Local CSV file if API is unavailable
   - **Production API Key**: Unlimited access (579b464db66ec23bdd0000015eb88b6f030349cb4f46c4631fb80919)
   - **Total Available**: ~3,000+ monitoring stations across India
   - **Batch Size**: Up to 1,000 records per API request
   - **Supported Pollutants**: PM10, PM2.5, NO2, SO2, CO, OZONE, NH3
   - **Coverage**: All major Indian states and cities
   - **Update Frequency**: Live data with 5-minute caching
   - **Configurable Limits**: Fetch 100-3000 records via sidebar slider

## Installation

1. Install Python 3.8 or higher
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure all three CSV files are in the same directory as `app.py`
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to `http://localhost:8501`

## Model Training

The app automatically trains two machine learning models:
- **RandomForestRegressor**: Predicts numeric AQI values
- **RandomForestClassifier**: Predicts AQI categories (Good, Satisfactory, Moderate, Poor, Very Poor, Severe)

Models are saved locally and reused on subsequent runs for better performance.

## AQI Categories (India CPCB Standard)

- **Good**: 0-50
- **Satisfactory**: 51-100
- **Moderate**: 101-200
- **Poor**: 201-300
- **Very Poor**: 301-400
- **Severe**: 401+

## Requirements

See `requirements.txt` for the complete list of Python dependencies.

## Features Overview

- **Interactive Visualizations**: Plotly charts and maps
- **Real-time Filtering**: Dynamic data filtering by location
- **Machine Learning**: Automated model training and prediction
- **Responsive Design**: Clean Streamlit interface
- **Data Processing**: Automatic data cleaning and feature engineering