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
    tab1, tab2, tab3 = st.tabs(['📊 Dashboard', '🤖 Prediction Tool', '🌍 Real-Time Insights'])

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
    # Weather-Enhanced AQI Prediction Tool Tab
    # ----------------------
    with tab2:
        st.header('🌤️ Weather-Enhanced AQI Prediction Tool')
        st.markdown('**Predict AQI using pollutant levels and real-time weather data**')
        
        if model_available:
            st.success("🎯 Using Advanced Weather-Enhanced ML Model (R² = 0.951)")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📍 Location & Date")
                
                # City selection
                available_cities = ['Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal', 
                                  'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam',
                                  'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kochi', 
                                  'Kolkata', 'Lucknow', 'Mumbai', 'Patna', 'Pune', 'Rajkot', 'Shillong', 
                                  'Thiruvananthapuram', 'Visakhapatnam']
                
                selected_city = st.selectbox('Select City', available_cities, index=available_cities.index('Delhi'))
                
                # Date selection (can predict future dates up to 7 days)
                min_date = datetime.now().date() - timedelta(days=365)  # 1 year back
                max_date = datetime.now().date() + timedelta(days=7)    # 7 days forward
                selected_date = st.date_input('Select Date', 
                                            value=datetime.now().date(),
                                            min_value=min_date, 
                                            max_value=max_date)
                
                st.subheader("🧪 Pollutant Levels")
                col_left, col_right = st.columns(2)
                
                with col_left:
                    pm25 = st.number_input('PM2.5 (µg/m³)', min_value=0.0, max_value=500.0, value=50.0, step=1.0)
                    pm10 = st.number_input('PM10 (µg/m³)', min_value=0.0, max_value=600.0, value=80.0, step=1.0)
                    no = st.number_input('NO (µg/m³)', min_value=0.0, max_value=200.0, value=20.0, step=1.0)
                    no2 = st.number_input('NO2 (µg/m³)', min_value=0.0, max_value=200.0, value=40.0, step=1.0)
                    nox = st.number_input('NOx (µg/m³)', min_value=0.0, max_value=300.0, value=60.0, step=1.0)
                    nh3 = st.number_input('NH3 (µg/m³)', min_value=0.0, max_value=400.0, value=25.0, step=1.0)
                
                with col_right:
                    co = st.number_input('CO (mg/m³)', min_value=0.0, max_value=30.0, value=1.5, step=0.1)
                    so2 = st.number_input('SO2 (µg/m³)', min_value=0.0, max_value=400.0, value=30.0, step=1.0)
                    o3 = st.number_input('O3 (µg/m³)', min_value=0.0, max_value=300.0, value=80.0, step=1.0)
                    benzene = st.number_input('Benzene (µg/m³)', min_value=0.0, max_value=50.0, value=2.0, step=0.1)
                    toluene = st.number_input('Toluene (µg/m³)', min_value=0.0, max_value=200.0, value=5.0, step=0.1)
                    xylene = st.number_input('Xylene (µg/m³)', min_value=0.0, max_value=200.0, value=3.0, step=0.1)
                
                # Fetch weather data automatically
                if st.button('🔮 Predict AQI with Live Weather Data', type='primary'):
                    with st.spinner('Fetching live weather data and predicting...'):
                        
                        # Fetch weather data
                        weather_data = fetch_weather_data(selected_city, selected_date)
                        
                        # Prepare input data
                        pollutants = {
                            'PM2.5': pm25, 'PM10': pm10, 'NO': no, 'NO2': no2, 'NOx': nox, 'NH3': nh3,
                            'CO': co, 'SO2': so2, 'O3': o3, 'Benzene': benzene, 'Toluene': toluene, 'Xylene': xylene
                        }
                        
                        # Create input features
                        input_data = {}
                        
                        # Add pollutant data
                        for key, value in pollutants.items():
                            input_data[key] = value
                        
                        # Add weather data
                        for key, value in weather_data.items():
                            input_data[f'weather_{key}'] = value
                        
                        # Add time features
                        date_obj = pd.to_datetime(selected_date)
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
                        input_data['city_encoded'] = city_mapping.get(selected_city, 10)  # Default to Delhi
                        
                        # Create DataFrame with all required features
                        input_df = pd.DataFrame([input_data])
                        
                        # Ensure all required features are present
                        for col in aqi_features:
                            if col not in input_df.columns:
                                input_df[col] = 0
                        
                        # Select features in the same order as training
                        input_df = input_df[aqi_features].fillna(0)
                        
                        # Make prediction
                        predicted_aqi = aqi_model.predict(input_df)[0]
                        predicted_category = get_aqi_category(predicted_aqi)
                        
                        # Display results
                        st.success("✅ Prediction Complete!")
                        
                        result_col1, result_col2, result_col3 = st.columns(3)
                        
                        with result_col1:
                            st.metric('🎯 Predicted AQI', f'{predicted_aqi:.0f}', 
                                    delta=f'Category: {predicted_category}')
                        
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
                
            with col2:
                st.subheader("📈 Model Information")
                st.info("""
                **Weather-Enhanced AQI Model**
                
                ✅ **Accuracy**: R² = 0.951
                📉 **Error**: MAE = 13.57
                🌡️ **Weather Features**: 15 parameters
                🧪 **Pollutants**: 12 parameters
                📅 **Training Data**: 2015-2020
                🏙️ **Cities**: 26+ Indian cities
                
                **Top Features:**
                1. PM2.5 (83.2%)
                2. PM10 (10.6%)
                3. CO (1.1%)
                4. O3 (0.6%)
                5. Weather data (4.5%)
                """)
                
                # Weather info section
                if st.checkbox("🌤️ Show Weather Preview"):
                    preview_weather = fetch_weather_data(selected_city, selected_date)
                    if preview_weather:
                        st.write(f"**Weather for {selected_city} on {selected_date}:**")
                        st.write(f"🌡️ Temp: {preview_weather.get('temperature_2m_mean', 'N/A')}°C")
                        st.write(f"💧 Humidity: {preview_weather.get('relative_humidity_2m_mean', 'N/A')}%")
                        st.write(f"💨 Wind: {preview_weather.get('wind_speed_10m_mean', 'N/A')} km/h")
                        st.write(f"☔ Rain: {preview_weather.get('precipitation_sum', 'N/A')} mm")
        else:
            st.error("⚠️ Weather-Enhanced AQI Model not available!")
            st.warning("Please run the data processing pipeline first:")
            st.code("python data_processor.py", language="bash")
            st.info("This will clean the data, fetch weather information, and train the advanced model.")

    # ----------------------
    # Real-Time Insights Tab
    # ----------------------
    with tab3:
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

if __name__ == '__main__':
    main()
