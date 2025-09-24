import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title='India AQI Dashboard', layout='wide', initial_sidebar_state='collapsed')

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, accuracy_score, classification_report
import plotly.express as px
import requests
import json

# ----------------------
# 1. DATA PREPARATION
# ----------------------

# File paths
YEARLY_PATH = 'Year-wise Details of Air Quality Index (AQI) levels in DelhiNCR from 2022 to 2024.csv'
STATE_PATH = 'StateUTs-wise Details of the Air Quality Index (AQI) of the Country during 2021.csv'
REALTIME_PATH = 'Real time Air Quality Index from various locations.csv'

# API Configuration
API_BASE_URL = 'https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69'
API_KEY = '579b464db66ec23bdd0000015eb88b6f030349cb4f46c4631fb80919'

# Load weather-enhanced AQI model
@st.cache_resource
def load_weather_aqi_model():
    """Load the weather-enhanced AQI prediction model"""
    try:
        model = joblib.load('aqi_weather_model.joblib')
        features = joblib.load('aqi_weather_features.joblib')
        return model, features, True
    except:
        return None, None, False

# Weather data fetching functions
def get_city_coordinates(city):
    """Get approximate coordinates for Indian cities"""
    city_coords = {
        'Ahmedabad': (23.0225, 72.5714),
        'Aizawl': (23.7367, 92.7173),
        'Amaravati': (16.5062, 80.6480),
        'Amritsar': (31.6340, 74.8723),
        'Bengaluru': (12.9716, 77.5946),
        'Bhopal': (23.2599, 77.4126),
        'Brajrajnagar': (21.8245, 83.9186),
        'Chandigarh': (30.7333, 76.7794),
        'Chennai': (13.0827, 80.2707),
        'Coimbatore': (11.0168, 76.9558),
        'Delhi': (28.7041, 77.1025),
        'Ernakulam': (9.9312, 76.2673),
        'Gurugram': (28.4595, 77.0266),
        'Guwahati': (26.1445, 91.7362),
        'Hyderabad': (17.3850, 78.4867),
        'Jaipur': (26.9124, 75.7873),
        'Jorapokhar': (23.7957, 86.4304),
        'Kochi': (9.9312, 76.2673),
        'Kolkata': (22.5726, 88.3639),
        'Lucknow': (26.8467, 80.9462),
        'Mumbai': (19.0760, 72.8777),
        'Patna': (25.5941, 85.1376),
        'Pune': (18.5204, 73.8567),
        'Rajkot': (22.3039, 70.8022),
        'Shillong': (25.5788, 91.8933),
        'Thiruvananthapuram': (8.5241, 76.9366),
        'Visakhapatnam': (17.6868, 83.2185)
    }
    return city_coords.get(city, (28.7041, 77.1025))  # Default to Delhi

def fetch_weather_data(city, date):
    """Fetch weather data from Open Meteo API for a specific city and date"""
    lat, lon = get_city_coordinates(city)
    
    # Open Meteo API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Format date
    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': date_str,
        'end_date': date_str,
        'daily': [
            'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
            'relative_humidity_2m_max', 'relative_humidity_2m_min', 'relative_humidity_2m_mean',
            'precipitation_sum', 'rain_sum', 'snowfall_sum',
            'wind_speed_10m_max', 'wind_speed_10m_mean',
            'wind_direction_10m_dominant', 'wind_gusts_10m_max',
            'pressure_msl_mean', 'sunshine_duration'
        ],
        'timezone': 'Asia/Kolkata'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'daily' in data and data['daily']:
            weather_data = {}
            for key, values in data['daily'].items():
                if key != 'time' and values and len(values) > 0:
                    weather_data[key] = values[0]
                else:
                    weather_data[key] = None
            
            return weather_data
        else:
            return {}
    except Exception as e:
        st.warning(f"Could not fetch weather data: {e}")
        return {}

# Load the models at startup
aqi_model, aqi_features, model_available = load_weather_aqi_model()

# Helper functions for real-time prediction
def process_realtime_pollutant_data(df, use_average=True, city=None):
    """Process real-time API data to extract pollutant values with enhanced city-specific logic"""
    if df.empty:
        return None
    
    # Enhanced pollutant mapping (same as debug menu)
    pollutant_mapping = {
        'PM2.5': 'PM2.5', 'PM10': 'PM10', 'NO': 'NO', 'NO2': 'NO2', 
        'NOX': 'NOx', 'NH3': 'NH3', 'CO': 'CO', 'SO2': 'SO2', 'O3': 'O3',
        'BENZENE': 'Benzene', 'TOLUENE': 'Toluene', 'XYLENE': 'Xylene',
        'C6H6': 'Benzene', 'C7H8': 'Toluene', 'C8H10': 'Xylene'
    }
    
    pollutants = {}
    
    # Process the API response - each row is a different pollutant
    if 'pollutant_avg' in df.columns and 'pollutant_id' in df.columns:
        for _, record in df.iterrows():
            pollutant_id = str(record.get('pollutant_id', '')).strip()
            avg_value = record.get('pollutant_avg', record.get('avg_value', 0))
            
            # Convert to float
            try:
                avg_value = float(avg_value) if avg_value not in [None, '', 'NA', 'N/A'] else 0
            except (ValueError, TypeError):
                avg_value = 0
            
            # Map API pollutant names to our format
            mapped_pollutant = pollutant_mapping.get(pollutant_id.upper(), pollutant_id)
            if mapped_pollutant and avg_value > 0:
                if mapped_pollutant in pollutants:
                    # If we already have this pollutant, average the values
                    if use_average:
                        pollutants[mapped_pollutant] = (pollutants[mapped_pollutant] + avg_value) / 2
                    # If not using average, keep the first value
                else:
                    pollutants[mapped_pollutant] = avg_value
    
    # City-specific realistic defaults for missing pollutants (same as debug menu)
    city_defaults = {
        'Delhi': {'PM2.5': 150, 'PM10': 200, 'NO2': 45, 'SO2': 30},
        'Mumbai': {'PM2.5': 70, 'PM10': 120, 'NO2': 35, 'SO2': 25},
        'Kolkata': {'PM2.5': 90, 'PM10': 150, 'NO2': 40, 'SO2': 28},
        'Chennai': {'PM2.5': 50, 'PM10': 80, 'NO2': 25, 'SO2': 18},
        'Bengaluru': {'PM2.5': 55, 'PM10': 90, 'NO2': 30, 'SO2': 20},
        'Hyderabad': {'PM2.5': 65, 'PM10': 110, 'NO2': 35, 'SO2': 22},
        'Ahmedabad': {'PM2.5': 85, 'PM10': 140, 'NO2': 38, 'SO2': 26},
        'Pune': {'PM2.5': 60, 'PM10': 100, 'NO2': 32, 'SO2': 21},
        'Jaipur': {'PM2.5': 95, 'PM10': 160, 'NO2': 40, 'SO2': 24},
        'Lucknow': {'PM2.5': 110, 'PM10': 180, 'NO2': 42, 'SO2': 27},
        'Kanpur': {'PM2.5': 125, 'PM10': 190, 'NO2': 44, 'SO2': 29},
        'Nagpur': {'PM2.5': 75, 'PM10': 125, 'NO2': 33, 'SO2': 23}
    }
    
    # Get city-specific defaults or use Delhi as fallback
    defaults = city_defaults.get(city, city_defaults['Delhi']) if city else city_defaults['Delhi']
    
    # Create complete pollutant data with defaults for missing values
    pollutant_data = {
        'PM2.5': pollutants.get('PM2.5', defaults['PM2.5']),
        'PM10': pollutants.get('PM10', defaults['PM10']),
        'NO': pollutants.get('NO', 15),
        'NO2': pollutants.get('NO2', defaults['NO2']),
        'NOx': pollutants.get('NOx', defaults['NO2'] + 15),
        'NH3': pollutants.get('NH3', 10),
        'CO': pollutants.get('CO', 1.8),
        'SO2': pollutants.get('SO2', defaults['SO2']),
        'O3': pollutants.get('O3', 70),
        'Benzene': pollutants.get('Benzene', 3.5),
        'Toluene': pollutants.get('Toluene', 8),
        'Xylene': pollutants.get('Xylene', 6)
    }
    
    return pollutant_data

def get_default_weather_data():
    """Get default weather data for when API fails"""
    return {
        'temperature_2m_max': 32.0,
        'temperature_2m_min': 22.0,
        'temperature_2m_mean': 27.0,
        'relative_humidity_2m_max': 75.0,
        'relative_humidity_2m_min': 45.0,
        'relative_humidity_2m_mean': 60.0,
        'precipitation_sum': 0.0,
        'rain_sum': 0.0,
        'snowfall_sum': 0.0,
        'wind_speed_10m_max': 15.0,
        'wind_speed_10m_mean': 8.0,
        'wind_direction_10m_dominant': 180.0,
        'wind_gusts_10m_max': 20.0,
        'pressure_msl_mean': 1013.0,
        'sunshine_duration': 28800.0
    }

def sanity_check_aqi_prediction(pollutant_data, predicted_aqi):
    """Perform a sanity check on AQI prediction based on actual training data patterns"""
    issues = []
    warnings = []
    
    # Get key pollutant values
    pm25 = pollutant_data.get('PM2.5', 0)
    pm10 = pollutant_data.get('PM10', 0)
    no2 = pollutant_data.get('NO2', 0)
    so2 = pollutant_data.get('SO2', 0)
    
    # AQI estimation based on ACTUAL TRAINING DATA PATTERNS (not theoretical standards)
    # Training data analysis shows these actual relationships:
    if pm25 <= 30:
        expected_aqi_range = (23, 100)  # Based on training data min and patterns
    elif pm25 <= 60:
        expected_aqi_range = (80, 180)  # Training mean ~140 for avg PM2.5 ~61
    elif pm25 <= 90:
        expected_aqi_range = (150, 250)  # Interpolated from training patterns
    elif pm25 <= 120:
        expected_aqi_range = (200, 310)  # Training data: PM2.5 ~117-122 -> AQI ~225-310
    elif pm25 <= 160:
        expected_aqi_range = (280, 370)  # Training data: PM2.5 140-160 -> AQI ~308-335 (mean 322)
    elif pm25 <= 250:
        expected_aqi_range = (350, 500)  # High PM2.5 values from training
    else:
        expected_aqi_range = (450, 677)  # Training max AQI was 677
    
    # Use training data variance for tolerance (±50 AQI is reasonable based on std dev)
    lower_bound = expected_aqi_range[0] - 50
    upper_bound = expected_aqi_range[1] + 50
    
    if predicted_aqi < lower_bound:
        issues.append(f"🚨 CRITICAL: Predicted AQI ({predicted_aqi:.0f}) is much lower than training data patterns ({expected_aqi_range[0]}-{expected_aqi_range[1]}) for PM2.5 = {pm25}")
    elif predicted_aqi > upper_bound:
        issues.append(f"🚨 CRITICAL: Predicted AQI ({predicted_aqi:.0f}) is much higher than training data patterns ({expected_aqi_range[0]}-{expected_aqi_range[1]}) for PM2.5 = {pm25}")
    elif not (expected_aqi_range[0] <= predicted_aqi <= expected_aqi_range[1]):
        warnings.append(f"ℹ️ INFO: Predicted AQI ({predicted_aqi:.0f}) is outside typical range ({expected_aqi_range[0]}-{expected_aqi_range[1]}) for PM2.5 = {pm25}, but within training data variance")
    
    # Additional validation checks
    if pm10 > 0 and pm25 > 0 and pm10 < pm25:
        warnings.append(f"⚠️ PM10 ({pm10}) is lower than PM2.5 ({pm25}) - this is unusual but possible")
    
    if no2 > 200:
        warnings.append(f"⚠️ Very high NO2 level ({no2}) may contribute to elevated AQI")
    
    if so2 > 80:
        warnings.append(f"⚠️ High SO2 level ({so2}) may contribute to elevated AQI")
        
    # Validate against training data correlation (r=0.924)
    if pm25 > 0:
        # Rough linear relationship from training: AQI ≈ 2.14 * PM2.5 + 9.3 (approximate from correlation)
        estimated_aqi = 2.14 * pm25 + 9.3
        if abs(predicted_aqi - estimated_aqi) > 80:  # Allow larger deviation due to other factors
            warnings.append(f"ℹ️ Prediction differs from PM2.5-only estimate by {abs(predicted_aqi - estimated_aqi):.0f} AQI - other factors (weather/pollutants) likely influencing result")
    
    return {
        'issues': issues,
        'warnings': warnings,
        'expected_range': expected_aqi_range,
        'is_reasonable': len(issues) == 0,
        'training_data_based': True
    }

def make_realtime_prediction(pollutant_data, weather_data, city, date):
    """Make AQI prediction using real-time pollutant and weather data with enhanced debugging"""
    try:
        # Load the model and features
        model, features, available = load_weather_aqi_model()
        if not available:
            return None
        
        # Create input features
        input_data = {}
        
        # Add pollutant data
        for key, value in pollutant_data.items():
            input_data[key] = value
        
        # Add weather data with correct feature names
        weather_feature_mapping = {
            'T': 'weather_temperature_2m_mean',
            'RH': 'weather_relative_humidity_2m_mean', 
            'WS': 'weather_wind_speed_10m_mean',
            'WD': 'weather_wind_direction_10m_dominant',
            'RF': 'weather_precipitation_sum',
            'BP': 'weather_pressure_msl_mean'
        }
        
        for key, value in weather_data.items():
            mapped_key = weather_feature_mapping.get(key, f'weather_{key}')
            input_data[mapped_key] = value
        
        # Add time features
        date_obj = pd.to_datetime(date)
        input_data['year'] = date_obj.year
        input_data['month'] = date_obj.month
        input_data['day_of_year'] = date_obj.dayofyear
        season = ((date_obj.month%12 + 3)//3)
        season_map = {1: 0, 2: 1, 3: 2, 4: 3}  # Winter, Spring, Summer, Fall
        input_data['season_encoded'] = season_map[season]
        
        # Encode city
        city_mapping = {
            'Ahmedabad': 0, 'Aizawl': 1, 'Amaravati': 2, 'Amritsar': 3, 'Bengaluru': 4, 'Bhopal': 5,
            'Brajrajnagar': 6, 'Chandigarh': 7, 'Chennai': 8, 'Coimbatore': 9, 'Delhi': 10,
            'Ernakulam': 11, 'Gurugram': 12, 'Guwahati': 13, 'Hyderabad': 14, 'Jaipur': 15,
            'Jorapokhar': 16, 'Kochi': 17, 'Kolkata': 18, 'Lucknow': 19, 'Mumbai': 20,
            'Patna': 21, 'Pune': 22, 'Rajkot': 23, 'Shillong': 24, 'Thiruvananthapuram': 25,
            'Visakhapatnam': 26
        }
        input_data['city_encoded'] = city_mapping.get(city, 10)  # Default to Delhi
        
        # Create DataFrame with all required features
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select features in the same order as training
        input_df = input_df[features].fillna(0)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Perform sanity check
        sanity_result = sanity_check_aqi_prediction(pollutant_data, prediction)
        
        # Enhanced DEBUG: Log prediction details for troubleshooting
        debug_info = {
            'city': city,
            'city_encoded': input_data.get('city_encoded', 'N/A'),
            'key_pollutants': {
                'PM2.5': pollutant_data.get('PM2.5', 'N/A'),
                'PM10': pollutant_data.get('PM10', 'N/A'),
                'NO2': pollutant_data.get('NO2', 'N/A'),
                'SO2': pollutant_data.get('SO2', 'N/A')
            },
            'key_weather': {
                'Temperature': weather_data.get('T', 'N/A'),
                'Humidity': weather_data.get('RH', 'N/A'),
                'Wind_Speed': weather_data.get('WS', 'N/A')
            },
            'feature_stats': {
                'total_features': len(input_df.columns),
                'zero_features': (input_df.iloc[0] == 0).sum(),
                'non_zero_features': (input_df.iloc[0] != 0).sum()
            },
            'sanity_check': {
                'is_reasonable': sanity_result['is_reasonable'],
                'expected_range': sanity_result['expected_range'],
                'issues_count': len(sanity_result['issues']),
                'warnings_count': len(sanity_result['warnings'])
            }
        }
        
        # Log the debug info (will appear in Streamlit)
        if hasattr(st, 'session_state'):
            if 'prediction_debug' not in st.session_state:
                st.session_state.prediction_debug = []
            
            # Keep only the last 5 debug entries
            if len(st.session_state.prediction_debug) >= 5:
                st.session_state.prediction_debug = st.session_state.prediction_debug[-4:]
            
            st.session_state.prediction_debug.append({
                **debug_info,
                'predicted_aqi': round(prediction, 1),
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'sanity_issues': sanity_result['issues'],
                'sanity_warnings': sanity_result['warnings']
            })
        
        return prediction
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def get_city_info(city):
    """Get information about a city"""
    city_info = {
        'Delhi': {'lat': 28.70, 'lon': 77.10, 'region': 'North India', 'population': '32M'},
        'Mumbai': {'lat': 19.08, 'lon': 72.88, 'region': 'West India', 'population': '21M'},
        'Kolkata': {'lat': 22.57, 'lon': 88.36, 'region': 'East India', 'population': '15M'},
        'Chennai': {'lat': 13.08, 'lon': 80.27, 'region': 'South India', 'population': '11M'},
        'Bengaluru': {'lat': 12.97, 'lon': 77.59, 'region': 'South India', 'population': '13M'},
        'Hyderabad': {'lat': 17.39, 'lon': 78.49, 'region': 'South India', 'population': '10M'},
        'Ahmedabad': {'lat': 23.02, 'lon': 72.57, 'region': 'West India', 'population': '8M'},
        'Pune': {'lat': 18.52, 'lon': 73.86, 'region': 'West India', 'population': '7M'},
        'Jaipur': {'lat': 26.91, 'lon': 75.79, 'region': 'North India', 'population': '4M'},
        'Lucknow': {'lat': 26.85, 'lon': 80.95, 'region': 'North India', 'population': '3M'}
    }
    return city_info.get(city, {'lat': 28.70, 'lon': 77.10, 'region': 'India'})

# Weather data fetching functions for different time periods
def get_weather_for_date(city, date_obj):
    """Fetch weather data for a specific date (past, present, or future)"""
    try:
        # Get city coordinates
        city_coords = get_city_coordinates(city)
        lat, lon = city_coords
        
        today = datetime.now().date()
        target_date = date_obj.strftime('%Y-%m-%d')
        
        # Determine which API endpoint to use
        if date_obj < today:
            # Historical weather data (ERA5 Archive) - keep this for now as fallback
            url = "https://api.open-meteo.com/v1/forecast"  # Use forecast API for consistency
            params = {
                'latitude': lat,
                'longitude': lon,
                'daily': 'temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,wind_speed_10m_max,precipitation_sum',
                'past_days': 7  # Get past 7 days of data
            }
        else:
            # Current and future weather data (forecast API handles both)
            url = "https://api.open-meteo.com/v1/forecast"
            if date_obj == today:
                # For today, get current weather
                params = {
                    'latitude': lat,
                    'longitude': lon,
                    'current': 'temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure',
                    'daily': 'precipitation_sum'  # Get daily precipitation
                }
            else:
                # For future dates, get daily forecast
                params = {
                    'latitude': lat,
                    'longitude': lon,
                    'daily': 'temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,wind_speed_10m_max,precipitation_sum',
                    'forecast_days': 7  # Get up to 7 days of forecast
                }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if date_obj == today and 'current' in data:
                # Current weather
                current = data.get('current', {})
                daily = data.get('daily', {})
                weather_data = {
                    'T': current.get('temperature_2m', 25.0),
                    'RH': current.get('relative_humidity_2m', 65.0),
                    'WS': current.get('wind_speed_10m', 5.0),
                    'WD': 180.0,  # Default wind direction
                    'RF': daily.get('precipitation_sum', [0.0])[0] if daily.get('precipitation_sum') else 0.0,
                    'BP': current.get('surface_pressure', 1013.0)
                }
            else:
                # Daily weather (historical or forecast)
                daily = data.get('daily', {})
                if daily and len(daily.get('time', [])) > 0:
                    # Find the index for our target date
                    target_index = 0
                    if 'time' in daily:
                        for i, date_str in enumerate(daily['time']):
                            if date_str.startswith(target_date):
                                target_index = i
                                break
                    
                    # Extract weather data for the target date
                    temp_max = daily.get('temperature_2m_max', [25.0])
                    temp_min = daily.get('temperature_2m_min', [20.0])
                    temp_mean = daily.get('temperature_2m_mean', [(temp_max[target_index] + temp_min[target_index]) / 2 if len(temp_max) > target_index and len(temp_min) > target_index else 25.0])
                    
                    weather_data = {
                        'T': temp_mean[target_index] if len(temp_mean) > target_index else (
                            (temp_max[target_index] + temp_min[target_index]) / 2 if len(temp_max) > target_index and len(temp_min) > target_index else 25.0
                        ),
                        'RH': daily.get('relative_humidity_2m_max', [65.0])[target_index] if len(daily.get('relative_humidity_2m_max', [])) > target_index else 65.0,
                        'WS': daily.get('wind_speed_10m_max', [5.0])[target_index] if len(daily.get('wind_speed_10m_max', [])) > target_index else 5.0,
                        'WD': 180.0,  # Default wind direction
                        'RF': daily.get('precipitation_sum', [0.0])[target_index] if len(daily.get('precipitation_sum', [])) > target_index else 0.0,
                        'BP': 1013.0  # Default surface pressure for daily data
                    }
                else:
                    # Fallback to default values
                    weather_data = {
                        'T': 25.0, 'RH': 65.0, 'WS': 5.0, 'WD': 180.0, 'RF': 0.0, 'BP': 1013.0
                    }
            
            return weather_data
            
        else:
            st.warning(f"Weather API returned status {response.status_code}, using default values")
            return {'T': 25.0, 'RH': 65.0, 'WS': 5.0, 'WD': 180.0, 'RF': 0.0, 'BP': 1013.0}
            
    except Exception as e:
        st.warning(f"Error fetching weather data: {str(e)}, using default values")
        return {'T': 25.0, 'RH': 65.0, 'WS': 5.0, 'WD': 180.0, 'RF': 0.0, 'BP': 1013.0}

def format_weather_data_for_display(weather_data):
    """Convert weather data keys to readable format and filter out zero values"""
    readable_mapping = {
        'T': 'Temperature (°C)',
        'RH': 'Relative Humidity (%)',
        'WS': 'Wind Speed (km/h)',
        'WD': 'Wind Direction (°)',
        'RF': 'Rainfall (mm)',
        'BP': 'Atmospheric Pressure (hPa)'
    }
    
    formatted_data = {}
    for key, value in weather_data.items():
        clean_key = key.replace('weather_', '')
        readable_name = readable_mapping.get(clean_key, clean_key)
        
        # Only include non-zero values (except temperature which can be 0)
        if value != 0 or clean_key == 'T':
            if isinstance(value, float):
                formatted_data[readable_name] = round(value, 1)
            else:
                formatted_data[readable_name] = value
    
    return formatted_data

# Fetch real-time data from API
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_realtime_api_data(limit=1000, state=None, city=None, max_pages=50):
    """Fetch real-time AQI data from the government API with pagination"""
    all_records = []
    page_limit = min(limit, 1000)  # API might have max limit per request
    
    # With unlimited API key, we can request more records per call
    api_limit = min(1000, limit)  # Try to get up to 1000 per request
    total_requests = min(max_pages, (limit // api_limit) + 1)
    
    for page in range(total_requests):
        params = {
            'api-key': API_KEY,
            'format': 'json',
            'limit': api_limit,
            'offset': page * api_limit
        }
        
        # Add filters if specified
        if state:
            params['filters[state]'] = state.replace(' ', '_')
        if city:
            params['filters[city]'] = city
        
        try:
            response = requests.get(API_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'records' in data and data['records']:
                all_records.extend(data['records'])
                
                # Show pagination info
                if page == 0:  # Show info only on first request
                    total_available = data.get('total', 0)
                    records_fetched = len(data['records'])
                    st.info(f"📊 Total records available: {total_available:,} | Fetching up to {limit:,} records | Got {records_fetched} records per request")
                
                # Break if we've got enough records or no more data
                if len(all_records) >= limit or len(data['records']) < api_limit:
                    break
            else:
                break
                
        except requests.exceptions.RequestException as e:
            if page == 0:  # Only show error on first request
                st.error(f"Error fetching data from API: {e}")
            break
    
    if all_records:
        df = pd.DataFrame(all_records[:limit])  # Limit to requested number
        st.success(f"✅ Successfully fetched {len(df):,} records from API")
        return df
    else:
        st.warning("No records found in API response")
        # Fallback to CSV file if API fails
        try:
            return pd.read_csv(REALTIME_PATH)
        except:
            return pd.DataFrame()

# Helper: AQI category bins (India CPCB standard)
def get_aqi_category(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

# Load and clean yearly/state datasets
def load_yearly_data():
    df = pd.read_csv(YEARLY_PATH)
    # Try to standardize column names
    df.columns = [c.strip().replace('\u200b', '').replace('\ufeff', '') for c in df.columns]
    # Calculate total AQI based on good/bad days for visualization
    good_cols = [col for col in df.columns if 'Good Days' in col]
    bad_cols = [col for col in df.columns if 'Bad Days' in col]
    if good_cols and bad_cols:
        df['Total_Good_Days'] = df[good_cols].sum(axis=1)
        df['Total_Bad_Days'] = df[bad_cols].sum(axis=1)
        df['Total_Days'] = df['Total_Good_Days'] + df['Total_Bad_Days']
        df['Good_Percentage'] = (df['Total_Good_Days'] / df['Total_Days'] * 100).fillna(0)
    return df

def load_state_data():
    df = pd.read_csv(STATE_PATH)
    df.columns = [c.strip().replace('\u200b', '').replace('\ufeff', '') for c in df.columns]
    # Calculate metrics from good/bad days
    good_col = '2021 - Good Days (AQI - 0-200)'
    bad_col = '2021 - Bad Days (Poor/ Very-Poor/ Severe) (AQI>200)'
    if good_col in df.columns and bad_col in df.columns:
        df['Total_Days'] = df[good_col].fillna(0) + df[bad_col].fillna(0)
        df['Good_Percentage'] = (df[good_col].fillna(0) / df['Total_Days'] * 100).fillna(0)
        # Estimate AQI based on percentage of bad days
        df['Estimated_AQI'] = 50 + (df[bad_col].fillna(0) / df['Total_Days'] * 300).fillna(0)
    return df

# Load and clean real-time dataset
def load_realtime_data(use_api=True, state_filter=None, city_filter=None, limit=200):
    if use_api:
        df = fetch_realtime_api_data(limit=limit, state=state_filter, city=city_filter)
    else:
        df = pd.read_csv(REALTIME_PATH)
    
    if df.empty:
        return df
    
    df.columns = [c.strip().replace('\u200b', '').replace('\ufeff', '') for c in df.columns]
    
    # Parse timestamp
    if 'last_update' in df.columns:
        df['datetime'] = pd.to_datetime(df['last_update'], errors='coerce', dayfirst=True)
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
    
    # Convert AQI-related columns
    aqi_cols = ['pollutant_avg', 'avg_value', 'pollutant_min', 'min_value', 'pollutant_max', 'max_value']
    for col in aqi_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Use the best available column as AQI proxy
    if 'avg_value' in df.columns:
        df['AQI'] = df['avg_value']
    elif 'pollutant_avg' in df.columns:
        df['AQI'] = df['pollutant_avg']
    else:
        # Create a synthetic AQI from available data
        df['AQI'] = 100  # Default moderate value
    
    # Only add AQI_Category if AQI column exists and has valid data
    if 'AQI' in df.columns and not df['AQI'].isna().all():
        df['AQI_Category'] = df['AQI'].apply(get_aqi_category)
    
    # Add pollutant columns for modeling
    if 'pollutant_id' in df.columns:
        # Create dummy pollutant columns based on pollutant_id
        pollutant_dummies = pd.get_dummies(df['pollutant_id'], prefix='pollutant')
        df = pd.concat([df, pollutant_dummies], axis=1)
    
    return df

# Extract stats for yearly/state datasets
def extract_stats_yearly(df):
    stats = {}
    if 'Good_Percentage' in df.columns:
        stats['mean_good_percentage'] = df['Good_Percentage'].mean()
        stats['total_good_days'] = df['Total_Good_Days'].sum() if 'Total_Good_Days' in df.columns else 0
        stats['total_bad_days'] = df['Total_Bad_Days'].sum() if 'Total_Bad_Days' in df.columns else 0
    return stats

def extract_stats_state(df):
    stats = {}
    if 'State/UTs' in df.columns:
        if 'Estimated_AQI' in df.columns:
            stats['mean_aqi'] = df.groupby('State/UTs')['Estimated_AQI'].mean().sort_values(ascending=False)
        if 'Good_Percentage' in df.columns:
            stats['good_percentage'] = df.groupby('State/UTs')['Good_Percentage'].mean().sort_values(ascending=False)
    return stats

# ----------------------
# 2. MODELING
# ----------------------

# Prepare features for modeling
def prepare_features(df):
    # Use available columns for modeling
    feature_cols = []
    # Add numeric pollutant columns only
    for col in df.columns:
        if col in ['pollutant_min', 'pollutant_max', 'pollutant_avg']:
            feature_cols.append(col)
    # Add time features
    for col in ['hour', 'day', 'month']:
        if col in df.columns:
            feature_cols.append(col)
    # Add location features (encode as categorical) - only if they exist
    for col in ['city', 'state', 'station']:
        if col in df.columns and df[col].notna().any():
            df[col + '_encoded'] = df[col].astype('category').cat.codes
            feature_cols.append(col + '_encoded')
    # Add latitude/longitude if available
    for col in ['latitude', 'longitude']:
        if col in df.columns:
            feature_cols.append(col)
    # Add pollutant dummy columns (these are numeric)
    for col in df.columns:
        if col.startswith('pollutant_') and col not in ['pollutant_min', 'pollutant_max', 'pollutant_avg', 'pollutant_id']:
            feature_cols.append(col)
    
    # Ensure we have at least some features
    if not feature_cols:
        return pd.DataFrame(), []
    
    X = df[feature_cols].fillna(0)
    # Convert all columns to numeric, replacing any remaining string values with 0
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    return X, feature_cols

def train_and_save_models(df):
    # Check if we have enough data for modeling
    if 'AQI' not in df.columns or df['AQI'].isna().all():
        return {'error': 'No AQI data available for modeling'}
    
    # Regression: predict AQI
    X, feature_cols = prepare_features(df)
    y_reg = df['AQI'].fillna(df['AQI'].mean())
    y_clf = df['AQI_Category'].fillna('Moderate')
    
    # Check if we have enough features
    if len(feature_cols) == 0 or X.shape[0] < 10:
        return {'error': 'Insufficient data for modeling'}
    
    # Encode target for classification
    y_clf_enc = y_clf.astype('category').cat.codes
    
    # Split
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf_enc, test_size=0.2, random_state=42)
    
    # Regression model
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_reg_train)
    y_reg_pred = reg.predict(X_test)
    reg_r2 = r2_score(y_reg_test, y_reg_pred)
    reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)
    
    # Classification model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_clf_train)
    y_clf_pred = clf.predict(X_test)
    clf_f1 = f1_score(y_clf_test, y_clf_pred, average='weighted')
    clf_acc = accuracy_score(y_clf_test, y_clf_pred)
    
    # Save models
    joblib.dump(reg, 'aqi_regressor.joblib')
    joblib.dump(clf, 'aqi_classifier.joblib')
    joblib.dump(feature_cols, 'model_features.joblib')
    joblib.dump(list(y_clf.astype('category').cat.categories), 'aqi_categories.joblib')
    
    return {
        'reg_r2': reg_r2, 'reg_mae': reg_mae,
        'clf_f1': clf_f1, 'clf_acc': clf_acc,
        'feature_cols': feature_cols
    }

# ----------------------
# 3. STREAMLIT UI
# ----------------------

def main():
    st.title('🇮🇳 India Air Quality Index (AQI) Dashboard')
    st.markdown('---')
    
    # Sidebar for data source selection
    st.sidebar.header('Data Source Settings')
    use_api = st.sidebar.checkbox('Use Live API Data', value=True, help='Fetch real-time data from government API')
    
    if use_api:
        max_records = st.sidebar.slider('Max Records to Fetch', 100, 3100, 1000, 100, 
                                      help='With unlimited API key, you can fetch up to 3000+ records from all monitoring stations across India.')
    else:
        max_records = 1000
        st.sidebar.info('Using local CSV file for real-time data')

    # Load data
    yearly_df = load_yearly_data()
    state_df = load_state_data()
    
    # Load real-time data (with API option)
    with st.spinner('Loading real-time data...'):
        realtime_df = load_realtime_data(use_api=use_api, limit=max_records)

    # Extract stats
    yearly_stats = extract_stats_yearly(yearly_df)
    state_stats = extract_stats_state(state_df)

    # Train/load models
    if not model_available:
        # Try to use old models as fallback
        if not (os.path.exists('aqi_regressor.joblib') and os.path.exists('aqi_classifier.joblib')):
            model_metrics = train_and_save_models(realtime_df)
        else:
            model_metrics = {}
            try:
                reg = joblib.load('aqi_regressor.joblib')
                clf = joblib.load('aqi_classifier.joblib')
                feature_cols = joblib.load('model_features.joblib')
                aqi_categories = joblib.load('aqi_categories.joblib')
            except:
                model_metrics = {'error': 'Could not load old models'}
    else:
        model_metrics = {'weather_model_loaded': True}

    # Tabs for navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['📊 Dashboard', '🔮 Smart Prediction', '🧪 Manual Prediction', '🌍 Real-Time Insights', '🔧 Debug Center'])

    # ----------------------
    # Dashboard Tab
    # ----------------------
    with tab1:
        st.header('AQI Trends & Analysis')
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Delhi NCR Good vs Bad Days (2022-2024)')
            if 'Total_Good_Days' in yearly_df.columns and 'Total_Bad_Days' in yearly_df.columns:
                # Create visualization of good vs bad days by city
                city_data = yearly_df.groupby('City Name')[['Total_Good_Days', 'Total_Bad_Days']].sum().reset_index()
                fig = px.bar(city_data, x='City Name', y=['Total_Good_Days', 'Total_Bad_Days'], 
                           title='Good vs Bad Days by City', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            if 'Good_Percentage' in yearly_df.columns:
                st.subheader('Good Air Quality Percentage by City')
                fig2 = px.bar(yearly_df, x='City Name', y='Good_Percentage', 
                            title='Percentage of Good Air Quality Days')
                st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            st.subheader('State-wise AQI Analysis (2021)')
            if 'State/UTs' in state_df.columns:
                if 'Estimated_AQI' in state_df.columns:
                    # Group by state and show average estimated AQI
                    state_aqi = state_df.groupby('State/UTs')['Estimated_AQI'].mean().sort_values(ascending=False).head(10)
                    fig3 = px.bar(x=state_aqi.index, y=state_aqi.values, 
                                title='Top 10 States by Estimated AQI (2021)')
                    st.plotly_chart(fig3, use_container_width=True)
                
                if 'Good_Percentage' in state_df.columns:
                    st.subheader('Good Air Quality Days by State')
                    state_good = state_df.groupby('State/UTs')['Good_Percentage'].mean().sort_values(ascending=False).head(10)
                    st.bar_chart(state_good)
        
        st.subheader('Real-time Data Overview')
        if not realtime_df.empty:
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric('Total Monitoring Stations', len(realtime_df))
            with col4:
                st.metric('States Covered', realtime_df['state'].nunique())
            with col5:
                st.metric('Cities Covered', realtime_df['city'].nunique())

    # ----------------------
    # Real-Time AQI Prediction Tool Tab
    # ----------------------
    with tab2:
        st.header('🤖 AQI Prediction Tool')
        st.markdown('**Predict AQI using live pollutant data with historical, current, or forecast weather information**')
        
        if model_available:
            st.success("🎯 Using Advanced Weather-Enhanced ML Model (R² = 0.951)")
            st.info("🔄 This tool automatically fetches real-time pollutant data and combines it with weather data (historical, current, or forecast) for prediction")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📍 Location & Date Selection")
                
                # City selection - use cities from the real-time API
                available_cities = ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bengaluru', 'Hyderabad', 
                                  'Ahmedabad', 'Pune', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur',
                                  'Indore', 'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara', 'Ghaziabad',
                                  'Ludhiana', 'Agra', 'Nashik', 'Faridabad', 'Meerut', 'Rajkot',
                                  'Kalyan-Dombivali', 'Vasai-Virar', 'Varanasi', 'Srinagar', 'Aurangabad',
                                  'Dhanbad', 'Amritsar', 'Navi Mumbai', 'Allahabad', 'Ranchi', 'Howrah',
                                  'Coimbatore', 'Jabalpur', 'Gwalior', 'Vijayawada', 'Jodhpur', 'Madurai',
                                  'Raipur', 'Kota', 'Chandigarh', 'Guwahati']
                
                selected_city = st.selectbox('Select City', available_cities, index=0)
                selected_state = st.selectbox('Select State (Optional)', 
                                            ['Auto-detect'] + ['Maharashtra', 'Uttar Pradesh', 'Delhi', 'Karnataka', 
                                             'Tamil Nadu', 'West Bengal', 'Gujarat', 'Rajasthan', 'Bihar', 
                                             'Madhya Pradesh', 'Punjab', 'Haryana', 'Telangana'], index=0)
                
                # Date selection (7 days past to 7 days future)
                selected_date = st.date_input('Date', 
                                            value=datetime.now().date(),
                                            min_value=datetime.now().date() - timedelta(days=7), 
                                            max_value=datetime.now().date() + timedelta(days=7),
                                            help="Select any date within 7 days past or future for prediction")
                
                # Show prediction type based on selected date
                today = datetime.now().date()
                if selected_date < today:
                    prediction_type = "📊 Historical Prediction"
                    date_info = f"Using historical weather data for {selected_date}"
                elif selected_date > today:
                    prediction_type = "🔮 Future Prediction"
                    date_info = f"Using weather forecast data for {selected_date}"
                else:
                    prediction_type = "⚡ Real-Time Prediction"
                    date_info = f"Using current weather data for {selected_date}"
                
                st.info(f"{prediction_type}: {date_info}")
                
                st.subheader("⚙️ Prediction Settings")
                
                max_records = st.slider('Max monitoring stations to fetch', 10, 500, 50, 10,
                                      help='More stations = more comprehensive data but slower processing')
                
                use_average = st.checkbox('Use average values from multiple stations', value=True,
                                        help='Average pollutant values from all stations in the city for more stable predictions')
                
                # Dynamic prediction button based on date
                button_text = "🔮 Get AQI Prediction" if selected_date != today else "⚡ Get Real-Time AQI Prediction"
                if st.button(button_text, type='primary'):
                    # Dynamic spinner message based on date
                    if selected_date < today:
                        spinner_msg = f'🔄 Fetching pollutant data and historical weather for {selected_date}...'
                    elif selected_date > today:
                        spinner_msg = f'🔄 Fetching pollutant data and weather forecast for {selected_date}...'
                    else:
                        spinner_msg = '🔄 Fetching live pollutant data and current weather information...'
                    
                    with st.spinner(spinner_msg):
                        
                        # Fetch real-time pollutant data
                        state_filter = None if selected_state == 'Auto-detect' else selected_state
                        realtime_data = fetch_realtime_api_data(limit=max_records, 
                                                              state=state_filter, 
                                                              city=selected_city)
                        
                        if realtime_data.empty:
                            st.error("❌ No real-time pollutant data found for the selected city/state")
                            st.info("Try selecting a different city or increase the number of stations")
                        else:
                            st.success(f"✅ Found {len(realtime_data)} monitoring stations")
                            
                            # Process pollutant data with city-specific enhancements
                            pollutant_data = process_realtime_pollutant_data(realtime_data, use_average, selected_city)
                            
                            if pollutant_data is None:
                                st.error("❌ Unable to extract pollutant values from the real-time data")
                                st.info("The API data might not contain the required pollutant measurements")
                            else:
                                # Show what pollutants were found from real API data
                                real_pollutants = []
                                if 'pollutant_id' in realtime_data.columns:
                                    unique_pollutants = realtime_data['pollutant_id'].dropna().unique()
                                    for pollutant in unique_pollutants:
                                        if str(pollutant).strip():
                                            real_pollutants.append(str(pollutant).strip())
                                
                                if real_pollutants:
                                    st.info(f"🧪 **Real pollutant data found**: {', '.join(real_pollutants)} ({len(real_pollutants)} types)")
                                else:
                                    st.warning("⚠️ Using city-specific default pollutant values")
                                # Fetch weather data for the selected date
                                weather_data = get_weather_for_date(selected_city, selected_date)
                                
                                if not weather_data:
                                    st.warning("⚠️ Unable to fetch weather data. Using default values.")
                                    weather_data = {'T': 25.0, 'RH': 65.0, 'WS': 5.0, 'WD': 180.0, 'RF': 0.0, 'BP': 1013.0}
                                
                                # Make prediction
                                predicted_aqi = make_realtime_prediction(pollutant_data, weather_data, 
                                                                       selected_city, selected_date)
                                
                                if predicted_aqi is not None:
                                    predicted_category = get_aqi_category(predicted_aqi)
                                    
                                    # Display results with dynamic message
                                    if selected_date < today:
                                        success_msg = "✅ Historical Analysis Complete!"
                                    elif selected_date > today:
                                        success_msg = "✅ Future Prediction Complete!"
                                    else:
                                        success_msg = "✅ Real-Time Prediction Complete!"
                                    
                                    st.success(success_msg)
                                    
                                    result_col1, result_col2, result_col3 = st.columns(3)
                                    
                                    with result_col1:
                                        st.metric('🎯 Predicted AQI', f'{predicted_aqi:.0f}')
                                    
                                    with result_col2:
                                        color_map = {
                                            'Good': '🟢', 'Satisfactory': '🟡', 'Moderate': '🟠', 
                                            'Poor': '🔴', 'Very Poor': '🟣', 'Severe': '⚫'
                                        }
                                        st.metric('📊 Air Quality', f'{color_map.get(predicted_category, "❔")} {predicted_category}')
                                    
                                    with result_col3:
                                        if predicted_aqi <= 50:
                                            health_msg = "Minimal Impact"
                                        elif predicted_aqi <= 100:
                                            health_msg = "Minor Breathing Discomfort"
                                        elif predicted_aqi <= 200:
                                            health_msg = "Breathing Discomfort"
                                        elif predicted_aqi <= 300:
                                            health_msg = "Respiratory Illness"
                                        elif predicted_aqi <= 400:
                                            health_msg = "Respiratory Effects"
                                        else:
                                            health_msg = "Emergency Conditions"
                                        st.metric('🏥 Health Impact', health_msg)
                                    
                                    # Show data sources
                                    st.subheader("📊 Data Sources Used")
                                    
                                    # Pollutant data details
                                    with st.expander("🧪 Real-Time Pollutant Data"):
                                        st.write(f"**Data from {len(realtime_data)} monitoring stations in {selected_city}**")
                                        
                                        # Show pollutant values used
                                        pollutant_df = pd.DataFrame([pollutant_data])
                                        st.dataframe(pollutant_df, use_container_width=True)
                                        
                                        # Show source data sample
                                        if len(realtime_data) > 0:
                                            st.write("**Sample monitoring station data:**")
                                            display_cols = ['station', 'pollutant_id', 'pollutant_avg', 'last_update']
                                            available_display_cols = [col for col in display_cols if col in realtime_data.columns]
                                            if available_display_cols:
                                                st.dataframe(realtime_data[available_display_cols].head(5), use_container_width=True)
                                    
                                    # Weather data details
                                    with st.expander("🌤️ Weather Data"):
                                        formatted_weather = format_weather_data_for_display(weather_data)
                                        weather_df = pd.DataFrame([formatted_weather])
                                        st.dataframe(weather_df, use_container_width=True)
                                    
                                    # Debug section to investigate AQI prediction discrepancies
                                    if hasattr(st, 'session_state') and hasattr(st.session_state, 'prediction_debug') and st.session_state.prediction_debug:
                                        with st.expander("🔍 Prediction Debug Info (Click to investigate AQI calculation)"):
                                            latest_debug = st.session_state.prediction_debug[-1]  # Get the most recent debug info
                                            
                                            st.write("**🎯 Prediction Analysis:**")
                                            debug_col1, debug_col2 = st.columns(2)
                                            
                                            with debug_col1:
                                                st.write("**📊 Key Inputs:**")
                                                st.write(f"🏙️ City: {latest_debug['city']} (encoded: {latest_debug['city_encoded']})")
                                                st.write(f"🧪 PM2.5: {latest_debug['key_pollutants']['PM2.5']}")
                                                st.write(f"🧪 PM10: {latest_debug['key_pollutants']['PM10']}")
                                                st.write(f"🧪 NO2: {latest_debug['key_pollutants']['NO2']}")
                                                st.write(f"🧪 SO2: {latest_debug['key_pollutants']['SO2']}")
                                                
                                            with debug_col2:
                                                st.write("**🌤️ Weather & Features:**")
                                                st.write(f"🌡️ Temperature: {latest_debug['key_weather']['Temperature']}°C")
                                                st.write(f"💧 Humidity: {latest_debug['key_weather']['Humidity']}%")
                                                st.write(f"💨 Wind Speed: {latest_debug['key_weather']['Wind_Speed']} km/h")
                                                st.write(f"📊 Total Features: {latest_debug['feature_stats']['total_features']}")
                                                st.write(f"⭕ Zero Features: {latest_debug['feature_stats']['zero_features']}")
                                            
                                            st.write(f"**🎯 Final Prediction: {latest_debug['predicted_aqi']} AQI**")
                                            
                                            # Show sanity check results
                                            st.write("**🔍 Sanity Check Results:**")
                                            if latest_debug.get('sanity_check', {}).get('is_reasonable', False):
                                                st.success(f"✅ Prediction appears reasonable for PM2.5 = {latest_debug['key_pollutants']['PM2.5']}")
                                                st.write(f"Expected range: {latest_debug['sanity_check']['expected_range'][0]}-{latest_debug['sanity_check']['expected_range'][1]} AQI")
                                            else:
                                                st.error(f"🚨 Prediction may be unreasonable for PM2.5 = {latest_debug['key_pollutants']['PM2.5']}")
                                                st.write(f"Expected range: {latest_debug['sanity_check']['expected_range'][0]}-{latest_debug['sanity_check']['expected_range'][1]} AQI")
                                            
                                            # Show issues and warnings from sanity check
                                            if 'sanity_issues' in latest_debug and latest_debug['sanity_issues']:
                                                st.error("**🚨 Critical Issues:**")
                                                for issue in latest_debug['sanity_issues']:
                                                    st.write(f"• {issue}")
                                            
                                            if 'sanity_warnings' in latest_debug and latest_debug['sanity_warnings']:
                                                st.warning("**⚠️ Warnings:**")
                                                for warning in latest_debug['sanity_warnings']:
                                                    st.write(f"• {warning}")
                                            
                                            # Additional feature analysis
                                            if latest_debug['feature_stats']['zero_features'] > latest_debug['feature_stats']['total_features'] * 0.7:
                                                st.info("ℹ️ >70% of features are zero - prediction relies heavily on available data")
                                            
                                            st.write(f"*Debug timestamp: {latest_debug['timestamp']}*")
                                
                                else:
                                    st.error("❌ Unable to make prediction. Please try again.")
                
            with col2:
                st.subheader("📈 Model Information")
                st.info("""
                **Real-Time AQI Prediction**
                
                ✅ **Accuracy**: R² = 0.951
                📉 **Error**: MAE = 13.57
                🔄 **Data Sources**: 
                - Government API (Pollutants)
                - Open Meteo API (Weather)
                
                📊 **Training Data**: 6,236 real measurements
                🧪 **PM2.5-AQI Correlation**: 0.924 (very strong)
                � **Predictions based on actual data patterns**
                📅 **Training Period**: 2015-2020
                🏙️ **Cities**: 26+ Indian cities
                
                💡 **Important**: AQI values reflect real-world 
                measurements from Indian monitoring stations.
                Example: PM2.5 = 150 → AQI ≈ 320 (actual data)
                
                **Key Features:**
                1. Real-time pollutant data
                2. Live weather integration  
                3. Multi-station averaging
                4. Training data validated predictions
                """)
                
                # Quick city stats
                if st.checkbox("📍 Show City Information"):
                    city_info = get_city_info(selected_city)
                    if city_info:
                        st.write(f"**{selected_city} Information:**")
                        st.write(f"🗺️ Coordinates: {city_info['lat']:.2f}, {city_info['lon']:.2f}")
                        st.write(f"🌍 Region: {city_info['region']}")
                        if 'population' in city_info:
                            st.write(f"👥 Population: {city_info['population']}")
        else:
            st.error("⚠️ Weather-Enhanced AQI Model not available!")
            st.warning("Please run the data processing pipeline first:")
            st.code("python retrain_model.py", language="bash")
            st.info("This will load the pre-trained model with weather data integration.")

    # ----------------------
    # Manual AQI Prediction Tab
    # ----------------------
    with tab3:
        st.header('🧪 Manual AQI Prediction Tool')
        st.markdown('**Enter specific pollutant values for custom AQI prediction**')
        
        if model_available:
            st.info("🎯 For users who want to input specific pollutant concentrations")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📍 Location & Date")
                
                # City selection
                available_cities = ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Bengaluru', 'Hyderabad', 
                                  'Ahmedabad', 'Pune', 'Jaipur', 'Lucknow', 'Kanpur', 'Nagpur']
                
                selected_city = st.selectbox('Select City', available_cities, index=0, key="manual_city")
                selected_date = st.date_input('Select Date', 
                                            value=datetime.now().date(),
                                            min_value=datetime.now().date() - timedelta(days=7), 
                                            max_value=datetime.now().date() + timedelta(days=7),
                                            key="manual_date")
                
                st.subheader("🧪 Pollutant Concentrations")
                col_left, col_right = st.columns(2)
                
                with col_left:
                    pm25 = st.number_input('PM2.5 (µg/m³)', min_value=0.0, max_value=500.0, value=50.0, step=1.0, key="manual_pm25")
                    pm10 = st.number_input('PM10 (µg/m³)', min_value=0.0, max_value=600.0, value=80.0, step=1.0, key="manual_pm10")
                    no = st.number_input('NO (µg/m³)', min_value=0.0, max_value=200.0, value=20.0, step=1.0, key="manual_no")
                    no2 = st.number_input('NO2 (µg/m³)', min_value=0.0, max_value=200.0, value=40.0, step=1.0, key="manual_no2")
                    nox = st.number_input('NOx (µg/m³)', min_value=0.0, max_value=300.0, value=60.0, step=1.0, key="manual_nox")
                    nh3 = st.number_input('NH3 (µg/m³)', min_value=0.0, max_value=400.0, value=25.0, step=1.0, key="manual_nh3")
                
                with col_right:
                    co = st.number_input('CO (mg/m³)', min_value=0.0, max_value=30.0, value=1.5, step=0.1, key="manual_co")
                    so2 = st.number_input('SO2 (µg/m³)', min_value=0.0, max_value=400.0, value=30.0, step=1.0, key="manual_so2")
                    o3 = st.number_input('O3 (µg/m³)', min_value=0.0, max_value=300.0, value=80.0, step=1.0, key="manual_o3")
                    benzene = st.number_input('Benzene (µg/m³)', min_value=0.0, max_value=50.0, value=2.0, step=0.1, key="manual_benzene")
                    toluene = st.number_input('Toluene (µg/m³)', min_value=0.0, max_value=200.0, value=5.0, step=0.1, key="manual_toluene")
                    xylene = st.number_input('Xylene (µg/m³)', min_value=0.0, max_value=200.0, value=3.0, step=0.1, key="manual_xylene")
                
                # Manual prediction button
                if st.button('🔮 Predict AQI with Manual Input', type='primary'):
                    with st.spinner('Fetching weather data and making prediction...'):
                        
                        # Fetch weather data for the selected date
                        weather_data = get_weather_for_date(selected_city, selected_date)
                        
                        if not weather_data:
                            st.warning("⚠️ Using default weather values")
                            weather_data = {'T': 25.0, 'RH': 65.0, 'WS': 5.0, 'WD': 180.0, 'RF': 0.0, 'BP': 1013.0}
                        
                        # Prepare pollutant data
                        pollutant_data = {
                            'PM2.5': pm25, 'PM10': pm10, 'NO': no, 'NO2': no2, 'NOx': nox, 'NH3': nh3,
                            'CO': co, 'SO2': so2, 'O3': o3, 'Benzene': benzene, 'Toluene': toluene, 'Xylene': xylene
                        }
                        
                        # Make prediction
                        predicted_aqi = make_realtime_prediction(pollutant_data, weather_data, 
                                                               selected_city, selected_date)
                        
                        if predicted_aqi is not None:
                            predicted_category = get_aqi_category(predicted_aqi)
                            
                            # Display results
                            st.success("✅ Manual Prediction Complete!")
                            
                            result_col1, result_col2, result_col3 = st.columns(3)
                            
                            with result_col1:
                                st.metric('🎯 Predicted AQI', f'{predicted_aqi:.0f}')
                            
                            with result_col2:
                                color_map = {
                                    'Good': '🟢', 'Satisfactory': '🟡', 'Moderate': '🟠', 
                                    'Poor': '🔴', 'Very Poor': '🟣', 'Severe': '⚫'
                                }
                                st.metric('📊 Air Quality', f'{color_map.get(predicted_category, "❔")} {predicted_category}')
                            
                            with result_col3:
                                if predicted_aqi <= 50:
                                    health_msg = "Minimal Impact"
                                elif predicted_aqi <= 100:
                                    health_msg = "Minor Breathing Discomfort"
                                elif predicted_aqi <= 200:
                                    health_msg = "Breathing Discomfort"
                                elif predicted_aqi <= 300:
                                    health_msg = "Respiratory Illness"
                                elif predicted_aqi <= 400:
                                    health_msg = "Respiratory Effects"
                                else:
                                    health_msg = "Emergency Conditions"
                                st.metric('🏥 Health Impact', health_msg)
                            
                            # Show input summary
                            st.subheader("📊 Input Summary")
                            
                            with st.expander("🧪 Pollutant Concentrations Used"):
                                pollutant_df = pd.DataFrame([pollutant_data])
                                st.dataframe(pollutant_df, use_container_width=True)
                            
                            with st.expander("🌤️ Weather Data Used"):
                                formatted_weather = format_weather_data_for_display(weather_data)
                                weather_df = pd.DataFrame([formatted_weather])
                                st.dataframe(weather_df, use_container_width=True)
                
            with col2:
                st.subheader("📋 Typical Pollutant Ranges")
                st.markdown("""
                **Normal Urban Levels (µg/m³):**
                - **PM2.5**: 15-60
                - **PM10**: 50-150  
                - **NO2**: 20-80
                - **SO2**: 10-50
                - **O3**: 50-120
                - **CO**: 0.5-4.0 (mg/m³)
                
                **Hazardous Levels:**
                - **PM2.5**: >250
                - **PM10**: >350
                - **NO2**: >400
                - **SO2**: >500
                - **O3**: >240
                - **CO**: >15 (mg/m³)
                """)
                
                st.subheader("💡 Tips")
                st.info("""
                **For Accurate Predictions:**
                
                1. Use recent monitoring data
                2. Consider seasonal variations
                3. Account for local emission sources
                4. Check multiple pollutants
                5. Compare with nearby stations
                """)
        else:
            st.error("⚠️ Model not available!")
            st.info("Please ensure the weather-enhanced model is loaded.")

    with tab4:
        st.header('🌍 Real-Time AQI Insights')
        if use_api:
            st.success('📡 Live data from Government of India API')
        else:
            st.info('📄 Using local CSV file')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('🔄 Refresh Data'):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if use_api:
                st.metric('Data Source', 'Live API')
            else:
                st.metric('Data Source', 'Local CSV')
        with col3:
            if 'datetime' in realtime_df.columns:
                latest_time = realtime_df['datetime'].max()
                if pd.notna(latest_time):
                    st.metric('Latest Update', latest_time.strftime('%Y-%m-%d %H:%M'))
        
        st.markdown('---')
        
        # Filter by state/city with API support
        if not realtime_df.empty:
            # Available states from API documentation
            api_states = ['Maharashtra', 'Uttar Pradesh', 'Rajasthan', 'Delhi', 'Bihar', 
                         'TamilNadu', 'Karnataka', 'Madhya Pradesh', 'West Bengal', 'Odisha', 
                         'Gujarat', 'Telangana', 'Chhattisgarh', 'Andhra Pradesh', 'Assam', 
                         'Punjab', 'Kerala', 'Chandigarh', 'Uttarakhand', 'Haryana', 
                         'Meghalaya', 'Jharkhand', 'Himachal Pradesh', 'Sikkim', 
                         'Arunachal Pradesh', 'Tripura']
            
            state_options = sorted(realtime_df['state'].dropna().unique()) if not realtime_df.empty else api_states
            city_options = sorted(realtime_df['city'].dropna().unique()) if not realtime_df.empty else []
            
            col1, col2 = st.columns(2)
            with col1:
                sel_state = st.selectbox('Filter by State', ['All'] + list(state_options))
            with col2:
                sel_city = st.selectbox('Filter by City', ['All'] + list(city_options))
            
            # Apply API filters if using API
            if use_api and (sel_state != 'All' or sel_city != 'All'):
                with st.spinner('Fetching filtered data...'):
                    filtered_df = load_realtime_data(
                        use_api=True, 
                        state_filter=sel_state if sel_state != 'All' else None,
                        city_filter=sel_city if sel_city != 'All' else None,
                        limit=max_records
                    )
                df_view = filtered_df
            else:
                df_view = realtime_df.copy()
                if sel_state != 'All':
                    df_view = df_view[df_view['state'] == sel_state]
                if sel_city != 'All':
                    df_view = df_view[df_view['city'] == sel_city]
            
            # Show latest data
            st.subheader('📊 Latest Monitoring Data')
            if 'datetime' in df_view.columns:
                df_view = df_view.sort_values('datetime', ascending=False)
            
            # Display key columns
            display_cols = ['state', 'city', 'station', 'pollutant_id', 'pollutant_avg', 'last_update']
            available_cols = [col for col in display_cols if col in df_view.columns]
            
            if available_cols:
                st.dataframe(df_view[available_cols].head(50), use_container_width=True)
            else:
                st.dataframe(df_view.head(50), use_container_width=True)
            
            # Show metrics
            st.subheader('📈 Summary Metrics')
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                if 'AQI' in df_view.columns and not df_view['AQI'].isna().all():
                    st.metric('Mean AQI', f"{df_view['AQI'].mean():.1f}")
                elif 'pollutant_avg' in df_view.columns:
                    st.metric('Mean Pollutant', f"{df_view['pollutant_avg'].mean():.1f}")
            with col2:
                if 'AQI' in df_view.columns and not df_view['AQI'].isna().all():
                    st.metric('Max AQI', f"{df_view['AQI'].max():.0f}")
                elif 'pollutant_avg' in df_view.columns:
                    st.metric('Max Pollutant', f"{df_view['pollutant_avg'].max():.0f}")
            with col3:
                st.metric('Monitoring Stations', len(df_view))
            with col4:
                st.metric('States Covered', df_view['state'].nunique() if 'state' in df_view.columns else 0)
            with col5:
                st.metric('Cities Covered', df_view['city'].nunique() if 'city' in df_view.columns else 0)
            
            # Show pollutant distribution
            if 'pollutant_id' in df_view.columns:
                st.subheader('🧪 Pollutant Distribution')
                pollutant_counts = df_view['pollutant_id'].value_counts()
                fig_pollutants = px.pie(values=pollutant_counts.values, names=pollutant_counts.index, 
                                      title='Distribution of Monitored Pollutants')
                st.plotly_chart(fig_pollutants, use_container_width=True)
            
            # Show map if coordinates available
            if 'latitude' in df_view.columns and 'longitude' in df_view.columns:
                st.subheader('🗺️ Interactive Station Map')
                
                # Prepare map data with all available columns
                base_cols = ['latitude', 'longitude', 'state', 'city', 'station', 'pollutant_id', 'last_update']
                available_cols = [col for col in base_cols if col in df_view.columns]
                map_data = df_view[available_cols].copy()
                
                # Add pollutant value columns if available
                value_cols = ['avg_value', 'pollutant_avg', 'min_value', 'max_value', 'pollutant_min', 'pollutant_max']
                for col in value_cols:
                    if col in df_view.columns:
                        map_data[col] = df_view[col]
                
                # Fill missing columns with defaults
                for col in ['state', 'city', 'station', 'pollutant_id', 'last_update']:
                    if col not in map_data.columns:
                        map_data[col] = 'N/A'
                
                # Use the best available pollutant value column
                pollutant_value_col = None
                for col in ['avg_value', 'pollutant_avg']:
                    if col in map_data.columns:
                        pollutant_value_col = col
                        break
                
                map_data = map_data.dropna(subset=['latitude', 'longitude'])
                
                if not map_data.empty:
                    # Convert to numeric
                    map_data['latitude'] = pd.to_numeric(map_data['latitude'], errors='coerce')
                    map_data['longitude'] = pd.to_numeric(map_data['longitude'], errors='coerce')
                    
                    if pollutant_value_col:
                        map_data[pollutant_value_col] = pd.to_numeric(map_data[pollutant_value_col], errors='coerce')
                    
                    map_data = map_data.dropna(subset=['latitude', 'longitude'])
                    
                    if not map_data.empty:
                        # Create hover data dictionary
                        hover_data = {
                            'state': True,
                            'city': True,
                            'pollutant_id': True,
                            'last_update': True,
                            'latitude': ':.4f',
                            'longitude': ':.4f'
                        }
                        
                        # Check if we have valid pollutant values for sizing and coloring
                        has_valid_values = False
                        if pollutant_value_col and pollutant_value_col in map_data.columns:
                            valid_values = map_data[pollutant_value_col].dropna()
                            if len(valid_values) > 0 and not valid_values.isna().all():
                                hover_data[pollutant_value_col] = ':.1f'
                                has_valid_values = True
                        
                        # Create interactive map with hover data
                        if has_valid_values:
                            # Use pollutant values for color and size
                            fig_map = px.scatter_mapbox(
                                map_data,
                                lat='latitude',
                                lon='longitude',
                                hover_name='station',
                                hover_data=hover_data,
                                color=pollutant_value_col,
                                size=map_data[pollutant_value_col].fillna(10),  # Fill NaN with default size
                                size_max=15,
                                zoom=4,
                                height=600,
                                title=f'Air Quality Monitoring Stations ({len(map_data)} stations)',
                                color_continuous_scale='Viridis'
                            )
                        else:
                            # Use pollutant type for color only
                            fig_map = px.scatter_mapbox(
                                map_data,
                                lat='latitude',
                                lon='longitude',
                                hover_name='station',
                                hover_data=hover_data,
                                color='pollutant_id',
                                zoom=4,
                                height=600,
                                title=f'Air Quality Monitoring Stations ({len(map_data)} stations)'
                            )
                        
                        # Update map style
                        fig_map.update_layout(
                            mapbox_style="open-street-map",
                            margin={"r": 0, "t": 50, "l": 0, "b": 0},
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_map, use_container_width=True)
                        
                        # Add legend explanation
                        if has_valid_values:
                            st.info("💡 **Map Legend**: Marker size and color represent pollutant levels. Hover over stations for detailed information including pollutant values, location, and last update time.")
                        else:
                            st.info("💡 **Map Legend**: Marker colors represent different pollutants. Hover over stations for detailed information including location, pollutant type, and last update time.")
                    else:
                        st.info('No valid coordinates available for mapping')
                else:
                    st.info('No station data available for mapping')
        else:
            st.warning('⚠️ No real-time data available. Please check your connection or try refreshing.')

    # Debug Center Tab
    with tab5:
        st.header('🔧 Debug Center')
        st.markdown('**Comprehensive testing and debugging tools for AQI prediction model**')
        
        if model_available:
            st.success("🎯 Model Available - Debug Tools Ready")
            
            # Debug options
            debug_col1, debug_col2 = st.columns([1, 1])
            
            with debug_col1:
                st.subheader("🧪 Model Testing")
                
                if st.button('🔍 Test All Cities', type='primary'):
                    st.subheader("📋 All Cities Prediction Test - **Using Real Data**")
                    
                    # List of all cities with coordinates
                    test_cities = {
                        'Delhi': {'lat': 28.6139, 'lon': 77.2090},
                        'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
                        'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
                        'Chennai': {'lat': 13.0827, 'lon': 80.2707},
                        'Bengaluru': {'lat': 12.9716, 'lon': 77.5946},
                        'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
                        'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714},
                        'Pune': {'lat': 18.5204, 'lon': 73.8567},
                        'Jaipur': {'lat': 26.9124, 'lon': 75.7873},
                        'Lucknow': {'lat': 26.8467, 'lon': 80.9462},
                        'Kanpur': {'lat': 26.4499, 'lon': 80.3319},
                        'Nagpur': {'lat': 21.1458, 'lon': 79.0882}
                    }
                    
                    results = []
                    debug_logs = []
                    
                    st.info("🌐 Fetching real-time data from APIs for each city...")
                    
                    for city, coords in test_cities.items():
                        with st.spinner(f'🔄 Fetching real data for {city}...'):
                            
                            # 1. Fetch real weather data from Open Meteo
                            weather_data = None
                            try:
                                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure&daily=precipitation_sum&timezone=auto"
                                weather_response = requests.get(weather_url, timeout=10)
                                
                                if weather_response.status_code == 200:
                                    weather_json = weather_response.json()
                                    current = weather_json.get('current', {})
                                    daily = weather_json.get('daily', {})
                                    
                                    weather_data = {
                                        'T': current.get('temperature_2m', 25),
                                        'RH': current.get('relative_humidity_2m', 65),
                                        'WS': current.get('wind_speed_10m', 6),
                                        'WD': current.get('wind_direction_10m', 180),
                                        'RF': daily.get('precipitation_sum', [0])[0] if daily.get('precipitation_sum') else 0,
                                        'BP': current.get('surface_pressure', 1013)
                                    }
                                    debug_logs.append(f"✅ {city}: Weather fetched - T:{weather_data['T']}°C, RH:{weather_data['RH']}%")
                                else:
                                    debug_logs.append(f"❌ {city}: Weather API error {weather_response.status_code}")
                                    
                            except Exception as e:
                                debug_logs.append(f"❌ {city}: Weather fetch error - {str(e)}")
                            
                            # 2. Fetch real pollutant data from government API  
                            pollutant_data = None
                            try:
                                # Temporarily suppress Streamlit messages for cleaner debug output
                                with st.spinner(f'Fetching pollutant data for {city}...'):
                                    # Use the existing function to fetch real-time data for this city
                                    old_write = st.write
                                    old_success = st.success
                                    old_info = st.info
                                    old_warning = st.warning
                                    old_error = st.error
                                    
                                    # Temporarily disable Streamlit output
                                    st.write = lambda *args, **kwargs: None
                                    st.success = lambda *args, **kwargs: None
                                    st.info = lambda *args, **kwargs: None
                                    st.warning = lambda *args, **kwargs: None
                                    st.error = lambda *args, **kwargs: None
                                    
                                    try:
                                        city_df = fetch_realtime_api_data(limit=50, city=city)
                                    finally:
                                        # Restore Streamlit output functions
                                        st.write = old_write
                                        st.success = old_success
                                        st.info = old_info
                                        st.warning = old_warning
                                        st.error = old_error
                                
                                if not city_df.empty and len(city_df) > 0:
                                    # Process the API response - each row is a different pollutant
                                    pollutants = {}
                                    
                                    for _, record in city_df.iterrows():
                                        pollutant_id = str(record.get('pollutant_id', '')).strip()
                                        avg_value = record.get('pollutant_avg', record.get('avg_value', 0))
                                        
                                        # Convert to float
                                        try:
                                            avg_value = float(avg_value) if avg_value not in [None, '', 'NA', 'N/A'] else 0
                                        except (ValueError, TypeError):
                                            avg_value = 0
                                        
                                        # Map API pollutant names to our format
                                        pollutant_mapping = {
                                            'PM2.5': 'PM2.5', 'PM10': 'PM10', 'NO': 'NO', 'NO2': 'NO2', 
                                            'NOX': 'NOx', 'NH3': 'NH3', 'CO': 'CO', 'SO2': 'SO2', 'O3': 'O3',
                                            'BENZENE': 'Benzene', 'TOLUENE': 'Toluene', 'XYLENE': 'Xylene',
                                            'C6H6': 'Benzene', 'C7H8': 'Toluene', 'C8H10': 'Xylene'
                                        }
                                        
                                        mapped_pollutant = pollutant_mapping.get(pollutant_id.upper(), pollutant_id)
                                        if mapped_pollutant and avg_value > 0:
                                            pollutants[mapped_pollutant] = avg_value
                                    
                                    # Create complete pollutant data with defaults for missing values
                                    if pollutants:
                                        # City-specific realistic defaults for missing pollutants
                                        city_defaults = {
                                            'Delhi': {'PM2.5': 150, 'PM10': 200, 'NO2': 45, 'SO2': 30},
                                            'Mumbai': {'PM2.5': 70, 'PM10': 120, 'NO2': 35, 'SO2': 25},
                                            'Kolkata': {'PM2.5': 90, 'PM10': 150, 'NO2': 40, 'SO2': 28},
                                            'Chennai': {'PM2.5': 50, 'PM10': 80, 'NO2': 25, 'SO2': 18},
                                            'Bengaluru': {'PM2.5': 55, 'PM10': 90, 'NO2': 30, 'SO2': 20},
                                            'Hyderabad': {'PM2.5': 65, 'PM10': 110, 'NO2': 35, 'SO2': 22},
                                            'Ahmedabad': {'PM2.5': 85, 'PM10': 140, 'NO2': 38, 'SO2': 26},
                                            'Pune': {'PM2.5': 60, 'PM10': 100, 'NO2': 32, 'SO2': 21},
                                            'Jaipur': {'PM2.5': 95, 'PM10': 160, 'NO2': 40, 'SO2': 24},
                                            'Lucknow': {'PM2.5': 110, 'PM10': 180, 'NO2': 42, 'SO2': 27},
                                            'Kanpur': {'PM2.5': 125, 'PM10': 190, 'NO2': 44, 'SO2': 29},
                                            'Nagpur': {'PM2.5': 75, 'PM10': 125, 'NO2': 33, 'SO2': 23}
                                        }
                                        
                                        defaults = city_defaults.get(city, city_defaults['Delhi'])
                                        
                                        pollutant_data = {
                                            'PM2.5': pollutants.get('PM2.5', defaults['PM2.5']),
                                            'PM10': pollutants.get('PM10', defaults['PM10']),
                                            'NO': pollutants.get('NO', 15),
                                            'NO2': pollutants.get('NO2', defaults['NO2']),
                                            'NOx': pollutants.get('NOx', defaults['NO2'] + 15),
                                            'NH3': pollutants.get('NH3', 10),
                                            'CO': pollutants.get('CO', 1.8),
                                            'SO2': pollutants.get('SO2', defaults['SO2']),
                                            'O3': pollutants.get('O3', 70),
                                            'Benzene': pollutants.get('Benzene', 3.5),
                                            'Toluene': pollutants.get('Toluene', 8),
                                            'Xylene': pollutants.get('Xylene', 6)
                                        }
                                        
                                        # Log what we found
                                        found_pollutants = [k for k, v in pollutants.items() if v > 0]
                                        debug_logs.append(f"✅ {city}: Real pollutant data found - {', '.join(found_pollutants)} (Total: {len(found_pollutants)} pollutants)")
                                        debug_logs.append(f"   PM2.5: {pollutant_data['PM2.5']:.1f}, PM10: {pollutant_data['PM10']:.1f}, NO2: {pollutant_data['NO2']:.1f}")
                                    else:
                                        debug_logs.append(f"⚠️ {city}: API returned data but no valid pollutant values found")
                                else:
                                    debug_logs.append(f"⚠️ {city}: API returned empty data, using defaults")
                                    
                            except Exception as e:
                                debug_logs.append(f"⚠️ {city}: Pollutant fetch error - {str(e)}, using defaults")
                            
                            # Use default values if API fetch failed
                            if pollutant_data is None:
                                # City-specific realistic defaults
                                city_defaults = {
                                    'Delhi': {'PM2.5': 150, 'PM10': 200, 'NO2': 45, 'SO2': 30},
                                    'Mumbai': {'PM2.5': 70, 'PM10': 120, 'NO2': 35, 'SO2': 25},
                                    'Kolkata': {'PM2.5': 90, 'PM10': 150, 'NO2': 40, 'SO2': 28},
                                    'Chennai': {'PM2.5': 50, 'PM10': 80, 'NO2': 25, 'SO2': 18},
                                    'Bengaluru': {'PM2.5': 55, 'PM10': 90, 'NO2': 30, 'SO2': 20},
                                    'Hyderabad': {'PM2.5': 65, 'PM10': 110, 'NO2': 35, 'SO2': 22},
                                    'Ahmedabad': {'PM2.5': 85, 'PM10': 140, 'NO2': 38, 'SO2': 26},
                                    'Pune': {'PM2.5': 60, 'PM10': 100, 'NO2': 32, 'SO2': 21},
                                    'Jaipur': {'PM2.5': 95, 'PM10': 160, 'NO2': 40, 'SO2': 24},
                                    'Lucknow': {'PM2.5': 110, 'PM10': 180, 'NO2': 42, 'SO2': 27},
                                    'Kanpur': {'PM2.5': 125, 'PM10': 190, 'NO2': 44, 'SO2': 29},
                                    'Nagpur': {'PM2.5': 75, 'PM10': 125, 'NO2': 33, 'SO2': 23}
                                }
                                
                                defaults = city_defaults.get(city, city_defaults['Delhi'])
                                pollutant_data = {
                                    'PM2.5': defaults['PM2.5'], 'PM10': defaults['PM10'],
                                    'NO': 15, 'NO2': defaults['NO2'], 'NOx': defaults['NO2'] + 15,
                                    'NH3': 10, 'CO': 1.8, 'SO2': defaults['SO2'], 'O3': 70,
                                    'Benzene': 3.5, 'Toluene': 8, 'Xylene': 6
                                }
                            
                            if weather_data is None:
                                # City-specific weather defaults
                                weather_defaults = {
                                    'Delhi': {'T': 28, 'RH': 60}, 'Mumbai': {'T': 30, 'RH': 75},
                                    'Chennai': {'T': 32, 'RH': 70}, 'Bengaluru': {'T': 25, 'RH': 65},
                                    'Kolkata': {'T': 29, 'RH': 80}, 'Hyderabad': {'T': 31, 'RH': 55}
                                }
                                defaults = weather_defaults.get(city, {'T': 27, 'RH': 65})
                                weather_data = {**defaults, 'WS': 6, 'WD': 180, 'RF': 0, 'BP': 1013}
                            
                            # 3. Make prediction with real data
                            pred = make_realtime_prediction(pollutant_data, weather_data, city, '2025-09-24')
                            
                            if pred is not None:
                                status = "✅ Real Data Success" if 'Real pollutant data' in str(debug_logs[-2:]) else "✅ Default Data Success"
                                results.append({
                                    'City': city,
                                    'Predicted_AQI': round(pred, 1),
                                    'Category': get_aqi_category(pred),
                                    'PM2.5': pollutant_data['PM2.5'],
                                    'Temperature': weather_data['T'],
                                    'Humidity': weather_data['RH'],
                                    'Status': status
                                })
                                debug_logs.append(f"🎯 {city}: Final AQI = {pred:.1f}")
                            else:
                                results.append({
                                    'City': city,
                                    'Predicted_AQI': 'Error',
                                    'Category': 'Error',
                                    'PM2.5': 'N/A',
                                    'Temperature': 'N/A', 
                                    'Humidity': 'N/A',
                                    'Status': '❌ Prediction Failed'
                                })
                                debug_logs.append(f"❌ {city}: Prediction failed")
                    
                    # Display results
                    if results:
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Download results as CSV
                        csv_data = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_data,
                            file_name=f"debug_city_test_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
                            mime="text/csv"
                        )
                        
                        # Statistics
                        successful_preds = [r for r in results if '✅' in r['Status'] and isinstance(r['Predicted_AQI'], (int, float))]
                        if successful_preds:
                            aqi_values = [float(r['Predicted_AQI']) for r in successful_preds]
                            
                            if aqi_values:
                                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                                with stats_col1:
                                    st.metric('🏙️ Cities Tested', len(test_cities))
                                with stats_col2:
                                    st.metric('📊 Min AQI', f"{min(aqi_values):.1f}")
                                with stats_col3:
                                    st.metric('📈 Max AQI', f"{max(aqi_values):.1f}")
                                with stats_col4:
                                    st.metric('🎯 Avg AQI', f"{sum(aqi_values)/len(aqi_values):.1f}")
                                
                                # Check for identical predictions
                                unique_predictions = len(set(aqi_values))
                                aqi_range = max(aqi_values) - min(aqi_values)
                                
                                if unique_predictions == 1:
                                    st.error(f"🚨 **PROBLEM DETECTED**: All cities returned identical AQI value ({aqi_values[0]:.1f})")
                                    st.info("This indicates an issue with the prediction model or feature processing.")
                                elif aqi_range < 5:
                                    st.error(f"🚨 **LOW VARIATION**: AQI range is only {aqi_range:.1f} points across all cities")
                                    st.info("This suggests the model isn't responding properly to different city/weather/pollution data.")
                                elif unique_predictions < len(aqi_values) * 0.3:  # Less than 30% unique
                                    st.warning(f"⚠️ **LIMITED VARIATION**: Only {unique_predictions} unique predictions out of {len(aqi_values)} cities")
                                else:
                                    st.success(f"✅ **GOOD VARIATION**: {unique_predictions} unique predictions, range: {aqi_range:.1f} AQI points")
                        
                        # Debug logs
                        with st.expander("🔍 Detailed Debug Logs"):
                            for log in debug_logs:
                                st.text(log)
                
                if st.button('🔬 Detailed Feature Analysis'):
                    st.subheader("🔬 Feature Analysis")
                    
                    # Load model details
                    model, features, available = load_weather_aqi_model()
                    if available:
                        st.success(f"📋 Model expects {len(features)} features")
                        
                        # Show feature names
                        with st.expander("📝 Expected Feature Names"):
                            feature_df = pd.DataFrame({'Feature_Name': features, 'Index': range(len(features))})
                            st.dataframe(feature_df, use_container_width=True)
                        
                        # Test feature creation for Delhi with real data
                        st.subheader("🧪 Feature Creation Test (Delhi with Real Data)")
                        
                        # Get real data for Delhi
                        try:
                            # Fetch real weather for Delhi
                            weather_url = "https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure&daily=precipitation_sum&timezone=auto"
                            weather_response = requests.get(weather_url, timeout=10)
                            
                            if weather_response.status_code == 200:
                                weather_json = weather_response.json()
                                current = weather_json.get('current', {})
                                daily = weather_json.get('daily', {})
                                
                                test_weather = {
                                    'T': current.get('temperature_2m', 26),
                                    'RH': current.get('relative_humidity_2m', 68),
                                    'WS': current.get('wind_speed_10m', 7),
                                    'WD': current.get('wind_direction_10m', 200),
                                    'RF': daily.get('precipitation_sum', [2])[0] if daily.get('precipitation_sum') else 2,
                                    'BP': current.get('surface_pressure', 1015)
                                }
                                st.success("✅ Using real weather data from Open Meteo")
                            else:
                                test_weather = {'T': 26, 'RH': 68, 'WS': 7, 'WD': 200, 'RF': 2, 'BP': 1015}
                                st.warning("⚠️ Using default weather data (API error)")
                                
                        except Exception as e:
                            test_weather = {'T': 26, 'RH': 68, 'WS': 7, 'WD': 200, 'RF': 2, 'BP': 1015}
                            st.warning(f"⚠️ Using default weather data (Error: {str(e)})")
                        
                        # Use realistic Delhi pollutant data
                        test_pollutants = {'PM2.5': 150, 'PM10': 200, 'NO': 25, 'NO2': 45, 'NOx': 70, 'NH3': 20, 'CO': 3.0, 'SO2': 30, 'O3': 85, 'Benzene': 6.0, 'Toluene': 12, 'Xylene': 9}
                        
                        # Show input data
                        st.subheader("📊 Input Data")
                        input_col1, input_col2 = st.columns(2)
                        
                        with input_col1:
                            st.text("🌡️ Weather Data:")
                            for key, value in test_weather.items():
                                st.text(f"  {key}: {value}")
                        
                        with input_col2:
                            st.text("🏭 Pollutant Data:")
                            for key, value in list(test_pollutants.items())[:6]:
                                st.text(f"  {key}: {value}")
                            st.text(f"  ... and {len(test_pollutants)-6} more")
                        
                        # Recreate the feature creation logic step by step
                        input_data = {}
                        
                        # Step 1: Add pollutant data
                        st.subheader("🔧 Step 1: Adding Pollutant Features")
                        for key, value in test_pollutants.items():
                            input_data[key] = value
                        st.text(f"Added {len(test_pollutants)} pollutant features")
                        
                        # Step 2: Add weather data with mapping
                        st.subheader("🔧 Step 2: Adding Weather Features")
                        weather_feature_mapping = {
                            'T': 'weather_temperature_2m_mean',
                            'RH': 'weather_relative_humidity_2m_mean', 
                            'WS': 'weather_wind_speed_10m_mean',
                            'WD': 'weather_wind_direction_10m_dominant',
                            'RF': 'weather_precipitation_sum',
                            'BP': 'weather_pressure_msl_mean'
                        }
                        
                        weather_mappings = []
                        for key, value in test_weather.items():
                            mapped_key = weather_feature_mapping.get(key, f'weather_{key}')
                            input_data[mapped_key] = value
                            weather_mappings.append(f"{key} → {mapped_key} = {value}")
                        
                        for mapping in weather_mappings:
                            st.text(f"  {mapping}")
                        
                        # Step 3: Add time features
                        st.subheader("🔧 Step 3: Adding Time Features")
                        date_obj = pd.to_datetime('2025-09-24')
                        input_data['year'] = date_obj.year
                        input_data['month'] = date_obj.month
                        input_data['day_of_year'] = date_obj.dayofyear
                        season = ((date_obj.month%12 + 3)//3)
                        season_map = {1: 0, 2: 1, 3: 2, 4: 3}
                        input_data['season_encoded'] = season_map[season]
                        
                        time_features = ['year', 'month', 'day_of_year', 'season_encoded']
                        for tf in time_features:
                            st.text(f"  {tf} = {input_data[tf]}")
                        
                        # Step 4: Add city encoding
                        st.subheader("🔧 Step 4: Adding City Encoding")
                        input_data['city_encoded'] = 10  # Delhi
                        st.text(f"  city_encoded = 10 (Delhi)")
                        
                        # Step 5: Create DataFrame
                        input_df = pd.DataFrame([input_data])
                        
                        # Step 6: Show created features vs expected
                        st.subheader("🔧 Step 5: Feature Matching Analysis")
                        
                        created_features = set(input_df.columns)
                        expected_features = set(features)
                        
                        missing_features = expected_features - created_features
                        extra_features = created_features - expected_features
                        matching_features = created_features & expected_features
                        
                        match_col1, match_col2, match_col3 = st.columns(3)
                        
                        with match_col1:
                            st.success(f"✅ **Matching**: {len(matching_features)}")
                            if len(matching_features) < 10:
                                with st.expander("See matching features"):
                                    for feature in sorted(matching_features):
                                        st.text(f"• {feature}")
                        
                        with match_col2:
                            if missing_features:
                                st.error(f"❌ **Missing**: {len(missing_features)}")
                                with st.expander("See missing features"):
                                    for feature in sorted(missing_features):
                                        st.text(f"• {feature}")
                            else:
                                st.success("✅ **No Missing Features**")
                        
                        with match_col3:
                            if extra_features:
                                st.warning(f"⚠️ **Extra**: {len(extra_features)}")
                                with st.expander("See extra features"):
                                    for feature in sorted(extra_features):
                                        st.text(f"• {feature}")
                            else:
                                st.info("ℹ️ **No Extra Features**")
                        
                        # Add missing features with zeros
                        for col in features:
                            if col not in input_df.columns:
                                input_df[col] = 0
                        
                        # Final feature vector
                        final_df = input_df[features].fillna(0)
                        
                        # Step 6: Show final feature statistics
                        st.subheader("🔧 Step 6: Final Feature Vector Analysis")
                        zero_features = (final_df.iloc[0] == 0).sum()
                        non_zero_features = len(features) - zero_features
                        
                        final_col1, final_col2, final_col3 = st.columns(3)
                        with final_col1:
                            st.metric("📊 Total Features", len(features))
                        with final_col2:
                            st.metric("✅ Non-Zero Features", non_zero_features)
                        with final_col3:
                            st.metric("⭕ Zero Features", zero_features)
                        
                        if zero_features > len(features) * 0.7:
                            st.error("🚨 **CRITICAL**: >70% features are zero - this will cause identical predictions!")
                        elif zero_features > len(features) * 0.5:
                            st.warning("⚠️ **WARNING**: >50% features are zero - predictions may lack variation")
                        else:
                            st.success("✅ **GOOD**: Most features have non-zero values")
                        
                        with st.expander("🎯 Final Feature Vector (Fed to Model)"):
                            final_features_df = final_df.T.reset_index()
                            final_features_df.columns = ['Feature', 'Value']
                            # Highlight non-zero features
                            final_features_df['Status'] = final_features_df['Value'].apply(lambda x: '✅ Non-Zero' if x != 0 else '⭕ Zero')
                            st.dataframe(final_features_df, use_container_width=True)
                        
                        # Make prediction with debug
                        prediction = model.predict(final_df)[0]
                        st.success(f"🎯 **Test Prediction**: {prediction:.2f} AQI ({get_aqi_category(prediction)})")
                        
                    else:
                        st.error("❌ Model not available for feature analysis")
                        
                        # Add missing features
                        for col in features:
                            if col not in input_df.columns:
                                input_df[col] = 0
                        
                        # Final feature vector
                        final_df = input_df[features].fillna(0)
                        
                        with st.expander("🎯 Final Feature Vector (Fed to Model)"):
                            final_features_df = final_df.T.reset_index()
                            final_features_df.columns = ['Feature', 'Value']
                            st.dataframe(final_features_df, use_container_width=True)
                        
                        # Make prediction with debug
                        prediction = model.predict(final_df)[0]
                        st.success(f"🎯 **Test Prediction**: {prediction:.2f} AQI")
                        
                        # Check for zero-heavy features
                        zero_features = (final_df.iloc[0] == 0).sum()
                        st.info(f"📊 Features set to zero: {zero_features} out of {len(features)}")
                        if zero_features > len(features) * 0.5:
                            st.warning("⚠️ More than 50% of features are zero - this might cause identical predictions!")
            
            with debug_col2:
                st.subheader("📊 Pollutant Variation Test")
                
                if st.button('🌡️ Test Pollutant Sensitivity'):
                    st.subheader("🌡️ Pollutant Sensitivity Analysis")
                    st.info("Testing how the model responds to different pollution levels using Delhi as test city")
                    
                    # Test with different pollutant levels
                    test_scenarios = {
                        'Clean Air': {'PM2.5': 15, 'PM10': 30, 'NO': 5, 'NO2': 10, 'NOx': 15, 'NH3': 3, 'CO': 0.5, 'SO2': 5, 'O3': 30, 'Benzene': 1.0, 'Toluene': 2, 'Xylene': 1},
                        'Moderate Pollution': {'PM2.5': 60, 'PM10': 100, 'NO': 15, 'NO2': 30, 'NOx': 45, 'NH3': 10, 'CO': 1.8, 'SO2': 20, 'O3': 70, 'Benzene': 3.5, 'Toluene': 8, 'Xylene': 6},
                        'High Pollution': {'PM2.5': 120, 'PM10': 200, 'NO': 25, 'NO2': 50, 'NOx': 75, 'NH3': 20, 'CO': 3.5, 'SO2': 40, 'O3': 120, 'Benzene': 8.0, 'Toluene': 15, 'Xylene': 12},
                        'Severe Pollution': {'PM2.5': 200, 'PM10': 350, 'NO': 40, 'NO2': 80, 'NOx': 120, 'NH3': 35, 'CO': 6.0, 'SO2': 70, 'O3': 180, 'Benzene': 15.0, 'Toluene': 25, 'Xylene': 20},
                        'Hazardous': {'PM2.5': 300, 'PM10': 500, 'NO': 60, 'NO2': 120, 'NOx': 180, 'NH3': 50, 'CO': 10.0, 'SO2': 100, 'O3': 250, 'Benzene': 25.0, 'Toluene': 40, 'Xylene': 30}
                    }
                    
                    # Get real weather for Delhi for consistent testing
                    try:
                        weather_url = "https://api.open-meteo.com/v1/forecast?latitude=28.6139&longitude=77.2090&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure&daily=precipitation_sum&timezone=auto"
                        weather_response = requests.get(weather_url, timeout=10)
                        
                        if weather_response.status_code == 200:
                            weather_json = weather_response.json()
                            current = weather_json.get('current', {})
                            daily = weather_json.get('daily', {})
                            
                            base_weather = {
                                'T': current.get('temperature_2m', 27),
                                'RH': current.get('relative_humidity_2m', 65),
                                'WS': current.get('wind_speed_10m', 6),
                                'WD': current.get('wind_direction_10m', 180),
                                'RF': daily.get('precipitation_sum', [0])[0] if daily.get('precipitation_sum') else 0,
                                'BP': current.get('surface_pressure', 1013)
                            }
                            st.success(f"✅ Using real Delhi weather: {base_weather['T']}°C, {base_weather['RH']}% humidity")
                        else:
                            base_weather = {'T': 27, 'RH': 65, 'WS': 6, 'WD': 180, 'RF': 0, 'BP': 1013}
                            st.warning("⚠️ Using default weather data")
                    except Exception as e:
                        base_weather = {'T': 27, 'RH': 65, 'WS': 6, 'WD': 180, 'RF': 0, 'BP': 1013}
                        st.warning(f"⚠️ Using default weather data (Error: {str(e)})")
                    
                    sensitivity_results = []
                    
                    for scenario_name, pollutants in test_scenarios.items():
                        with st.spinner(f'Testing {scenario_name}...'):
                            pred = make_realtime_prediction(pollutants, base_weather, 'Delhi', '2025-09-24')
                            if pred is not None:
                                sensitivity_results.append({
                                    'Scenario': scenario_name,
                                    'PM2.5': pollutants['PM2.5'],
                                    'PM10': pollutants['PM10'],
                                    'NO2': pollutants['NO2'],
                                    'SO2': pollutants['SO2'],
                                    'Predicted_AQI': round(pred, 1),
                                    'Category': get_aqi_category(pred),
                                    'Expected_Range': {
                                        'Clean Air': '0-50', 'Moderate Pollution': '51-100', 
                                        'High Pollution': '101-200', 'Severe Pollution': '201-300',
                                        'Hazardous': '301+'
                                    }[scenario_name]
                                })
                    
                    if sensitivity_results:
                        sens_df = pd.DataFrame(sensitivity_results)
                        st.dataframe(sens_df, use_container_width=True)
                        
                        # Analysis
                        aqi_values = [r['Predicted_AQI'] for r in sensitivity_results]
                        aqi_range = max(aqi_values) - min(aqi_values)
                        
                        sens_col1, sens_col2, sens_col3 = st.columns(3)
                        with sens_col1:
                            st.metric('📈 AQI Range', f"{aqi_range:.1f}")
                        with sens_col2:
                            st.metric('📊 Min AQI', f"{min(aqi_values):.1f}")
                        with sens_col3:
                            st.metric('📈 Max AQI', f"{max(aqi_values):.1f}")
                        
                        # Sensitivity assessment
                        if aqi_range < 10:
                            st.error("🚨 **VERY LOW SENSITIVITY**: Model shows almost no response to massive pollutant changes!")
                            st.info("Expected: Clean air (~25 AQI) to Hazardous (~400+ AQI) should show 300+ point difference")
                        elif aqi_range < 50:
                            st.warning("⚠️ **LOW SENSITIVITY**: Model shows limited response to pollutant changes")
                            st.info("The model should show much larger AQI differences between clean and hazardous air")
                        elif aqi_range < 100:
                            st.info("ℹ️ **MODERATE SENSITIVITY**: Model shows some response to pollutant changes")
                        else:
                            st.success("✅ **GOOD SENSITIVITY**: Model responds well to pollutant changes")
                        
                        # Check if predictions match expected ranges
                        st.subheader("📊 Expected vs Actual Analysis")
                        range_matches = 0
                        for result in sensitivity_results:
                            expected = result['Expected_Range']
                            actual = result['Predicted_AQI']
                            
                            if expected == '0-50' and actual <= 50:
                                range_matches += 1
                            elif expected == '51-100' and 51 <= actual <= 100:
                                range_matches += 1
                            elif expected == '101-200' and 101 <= actual <= 200:
                                range_matches += 1
                            elif expected == '201-300' and 201 <= actual <= 300:
                                range_matches += 1
                            elif expected == '301+' and actual > 300:
                                range_matches += 1
                        
                        match_percentage = (range_matches / len(sensitivity_results)) * 100
                        
                        if match_percentage >= 80:
                            st.success(f"✅ **EXCELLENT**: {match_percentage:.0f}% of predictions match expected ranges")
                        elif match_percentage >= 60:
                            st.info(f"ℹ️ **GOOD**: {match_percentage:.0f}% of predictions match expected ranges")
                        elif match_percentage >= 40:
                            st.warning(f"⚠️ **MODERATE**: {match_percentage:.0f}% of predictions match expected ranges")
                        else:
                            st.error(f"🚨 **POOR**: Only {match_percentage:.0f}% of predictions match expected ranges")
                    
                    else:
                        st.error("❌ No predictions could be made for sensitivity analysis")
                
                st.subheader("🌦️ Weather Impact Test")
                
                if st.button('☀️ Test Weather Sensitivity'):
                    st.subheader("☀️ Weather Sensitivity Analysis")
                    
                    # Test with different weather conditions
                    weather_scenarios = {
                        'Hot & Dry': {'T': 35, 'RH': 30, 'WS': 3, 'WD': 180, 'RF': 0, 'BP': 1010},
                        'Cool & Humid': {'T': 20, 'RH': 85, 'WS': 8, 'WD': 200, 'RF': 5, 'BP': 1020},
                        'Rainy': {'T': 25, 'RH': 90, 'WS': 12, 'WD': 220, 'RF': 15, 'BP': 1005},
                        'Windy': {'T': 28, 'RH': 50, 'WS': 20, 'WD': 270, 'RF': 0, 'BP': 1015}
                    }
                    
                    base_pollutants = {'PM2.5': 80, 'PM10': 120, 'NO': 18, 'NO2': 35, 'NOx': 50, 'NH3': 12, 'CO': 2.0, 'SO2': 22, 'O3': 75, 'Benzene': 4.0, 'Toluene': 9, 'Xylene': 7}
                    
                    weather_results = []
                    
                    for scenario_name, weather in weather_scenarios.items():
                        pred = make_realtime_prediction(base_pollutants, weather, 'Delhi', '2025-09-24')
                        if pred is not None:
                            weather_results.append({
                                'Scenario': scenario_name,
                                'Temperature': weather['T'],
                                'Humidity': weather['RH'],
                                'Wind_Speed': weather['WS'],
                                'Predicted_AQI': round(pred, 1),
                                'Category': get_aqi_category(pred)
                            })
                    
                    if weather_results:
                        weather_df = pd.DataFrame(weather_results)
                        st.dataframe(weather_df, use_container_width=True)
                        
                        # Check weather sensitivity
                        weather_aqi_range = max(weather_df['Predicted_AQI']) - min(weather_df['Predicted_AQI'])
                        st.metric('🌦️ Weather Impact Range', f"{weather_aqi_range:.1f}")
                        
                        if weather_aqi_range < 5:
                            st.warning("⚠️ Weather has minimal impact on predictions")
                        else:
                            st.success(f"✅ Weather shows {weather_aqi_range:.1f} point AQI variation")
                
                # Raw model info
                st.subheader("🔍 Model Information")
                model, features, available = load_weather_aqi_model()
                if available:
                    st.info(f"📋 Model Type: {type(model).__name__}")
                    st.info(f"🔢 Features Required: {len(features)}")
                    st.info(f"🎯 Model Available: {available}")
                else:
                    st.error("❌ Model not available")
                    
        else:
            st.error("❌ Model not available - cannot run debug tests")
            st.info("Please ensure the weather-enhanced model is loaded correctly")

if __name__ == '__main__':
    main()
