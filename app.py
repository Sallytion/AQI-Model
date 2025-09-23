import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
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
    st.set_page_config(page_title='India AQI Dashboard', layout='wide', initial_sidebar_state='collapsed')
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
    if not (os.path.exists('aqi_regressor.joblib') and os.path.exists('aqi_classifier.joblib')):
        model_metrics = train_and_save_models(realtime_df)
    else:
        model_metrics = {}
        reg = joblib.load('aqi_regressor.joblib')
        clf = joblib.load('aqi_classifier.joblib')
        feature_cols = joblib.load('model_features.joblib')
        aqi_categories = joblib.load('aqi_categories.joblib')

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
    # Prediction Tool Tab
    # ----------------------
    with tab2:
        st.header('AQI Prediction Tool')
        st.markdown('Enter pollutant levels and details to predict AQI:')
        
        # Check if models exist and were trained successfully
        if (os.path.exists('aqi_regressor.joblib') and os.path.exists('aqi_classifier.joblib') 
            and 'error' not in model_metrics):
            # Load models
            reg = joblib.load('aqi_regressor.joblib')
            clf = joblib.load('aqi_classifier.joblib')
            feature_cols = joblib.load('model_features.joblib')
            aqi_categories = joblib.load('aqi_categories.joblib')
            
            # User input widgets
            input_data = {}
            for col in feature_cols:
                if col in ['hour', 'day', 'month']:
                    input_data[col] = st.slider(col.capitalize(), 0, 23 if col=='hour' else 31 if col=='day' else 12, 1)
                elif '_encoded' in col:
                    st.write(f"{col.replace('_encoded', '')}: Encoded automatically")
                    input_data[col] = 0  # Default encoded value
                elif col in ['latitude', 'longitude']:
                    input_data[col] = st.number_input(col.capitalize(), value=28.6139 if col=='latitude' else 77.2090)
                else:
                    input_data[col] = st.number_input(col, min_value=0.0, max_value=1000.0, value=50.0)
            
            if st.button('Predict AQI'):
                X_input = np.array([list(input_data.values())])
                pred_aqi = reg.predict(X_input)[0]
                pred_cat_idx = clf.predict(X_input)[0]
                pred_cat = aqi_categories[pred_cat_idx] if pred_cat_idx < len(aqi_categories) else 'Unknown'
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Predicted AQI', f'{pred_aqi:.0f}')
                with col2:
                    st.metric('Predicted Category', pred_cat)
                
                # Feature importance
                st.subheader('Feature Importance')
                importances = reg.feature_importances_
                imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
                st.bar_chart(imp_df.set_index('feature'))
        else:
            st.warning('Models could not be trained due to insufficient data. Please check your CSV files.')
            if 'error' in model_metrics:
                st.error(f"Error: {model_metrics['error']}")

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
                st.subheader('🗺️ Station Locations')
                map_data = df_view[['latitude', 'longitude']].dropna()
                if not map_data.empty:
                    # Convert to numeric
                    map_data['latitude'] = pd.to_numeric(map_data['latitude'], errors='coerce')
                    map_data['longitude'] = pd.to_numeric(map_data['longitude'], errors='coerce')
                    map_data = map_data.dropna()
                    if not map_data.empty:
                        st.map(map_data)
                    else:
                        st.info('No valid coordinates available for mapping')
        else:
            st.warning('⚠️ No real-time data available. Please check your connection or try refreshing.')

if __name__ == '__main__':
    main()
