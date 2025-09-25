import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class AQIWeatherPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        
    def load_and_clean_data(self, file_path='2015-2020_data/city_day.csv'):
        """Load and clean the city_day.csv file by removing rows with missing values"""
        print("Loading city_day.csv...")
        df = pd.read_csv(file_path)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        # Remove rows with any missing values
        df_clean = df.dropna()
        print(f"After removing rows with missing values: {df_clean.shape}")
        
        # Convert Date to datetime
        df_clean['Date'] = pd.to_datetime(df_clean['Date'])
        
        # Save cleaned data
        df_clean.to_csv('cleaned_city_day.csv', index=False)
        print("Saved cleaned data to 'cleaned_city_day.csv'")
        
        return df_clean
    
    def get_city_coordinates(self, city):
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
    
    def fetch_weather_data(self, city, date, lat, lon):
        """Fetch weather data from Open Meteo API for a specific city and date"""
        # Open Meteo API endpoint
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Format date
        date_str = date.strftime('%Y-%m-%d')
        
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
                        weather_data[f'weather_{key}'] = values[0]
                    else:
                        weather_data[f'weather_{key}'] = None
                
                return weather_data
            else:
                return {}
        except Exception as e:
            print(f"Error fetching weather data for {city} on {date_str}: {e}")
            return {}
    
    def create_weather_enhanced_dataset(self, df_clean):
        """Create dataset with weather data for each city-date combination"""
        print("Fetching weather data from Open Meteo API...")
        
        weather_data_list = []
        
        # Get unique city-date combinations
        unique_combinations = df_clean[['City', 'Date']].drop_duplicates()
        total_combinations = len(unique_combinations)
        
        print(f"Total city-date combinations to process: {total_combinations}")
        
        for idx, (_, row) in enumerate(unique_combinations.iterrows()):
            if idx % 100 == 0:
                print(f"Progress: {idx}/{total_combinations} ({idx/total_combinations*100:.1f}%)")
            
            city = row['City']
            date = row['Date']
            lat, lon = self.get_city_coordinates(city)
            
            weather_data = self.fetch_weather_data(city, date, lat, lon)
            
            weather_row = {
                'City': city,
                'Date': date,
                **weather_data
            }
            weather_data_list.append(weather_row)
            
            # Add small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        print("Creating weather DataFrame...")
        weather_df = pd.DataFrame(weather_data_list)
        
        # Save weather data
        weather_df.to_csv('weather_data.csv', index=False)
        print("Saved weather data to 'weather_data.csv'")
        
        # Merge with original cleaned data
        print("Merging weather data with AQI data...")
        merged_df = pd.merge(df_clean, weather_df, on=['City', 'Date'], how='inner')
        
        # Save merged dataset
        merged_df.to_csv('aqi_weather_merged.csv', index=False)
        print(f"Saved merged dataset to 'aqi_weather_merged.csv' with shape: {merged_df.shape}")
        
        return merged_df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        feature_columns = []
        
        # Pollutant features
        pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        feature_columns.extend(pollutant_cols)
        
        # Weather features
        weather_cols = [col for col in df.columns if col.startswith('weather_')]
        feature_columns.extend(weather_cols)
        
        # Time features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['season'] = ((df['Date'].dt.month%12 + 3)//3).map({1:'Winter', 2:'Spring', 3:'Summer', 4:'Fall'})
        
        # Encode categorical features
        df['season_encoded'] = pd.Categorical(df['season']).codes
        df['city_encoded'] = pd.Categorical(df['City']).codes
        
        time_features = ['year', 'month', 'day_of_year', 'season_encoded', 'city_encoded']
        feature_columns.extend(time_features)
        
        # Select only available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].copy()
        
        # Fill missing values with median for numeric columns
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)
        
        return X, available_features
    
    def train_model(self, df):
        """Train Random Forest model to predict AQI"""
        print("Preparing features for model training...")
        
        # Prepare features
        X, feature_columns = self.prepare_features(df)
        y = df['AQI'].copy()
        
        print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
        print(f"Training data shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print("Training Random Forest model...")
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        self.feature_columns = feature_columns
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Feature Importance:")
        print(feature_importance.head(10))
        
        # Save model and features
        joblib.dump(self.model, 'aqi_weather_model.joblib')
        joblib.dump(feature_columns, 'aqi_weather_features.joblib')
        
        print("Model saved as 'aqi_weather_model.joblib'")
        print("Features saved as 'aqi_weather_features.joblib'")
        
        return r2, mae, feature_importance
    
    def predict_aqi(self, city, date, pollutants, weather_data):
        """Predict AQI for given city, date, pollutants, and weather data"""
        if self.model is None:
            try:
                self.model = joblib.load('aqi_weather_model.joblib')
                self.feature_columns = joblib.load('aqi_weather_features.joblib')
            except:
                raise ValueError("No trained model found. Please train the model first.")
        
        # Create input data
        input_data = {}
        
        # Universal default values for missing or zero pollutant data
        pollutant_defaults = {
            'PM2.5': 50.0, 'PM10': 80.0, 'NO': 20.0, 'NO2': 40.0, 
            'NOx': 60.0, 'NH3': 25.0, 'CO': 1.5, 'SO2': 30.0, 
            'O3': 80.0, 'Benzene': 2.0, 'Toluene': 5.0, 'Xylene': 3.0
        }
        
        # Add pollutant data with defaults for missing or zero values
        for col in pollutant_defaults.keys():
            value = pollutants.get(col, 0)
            # Use default if value is missing, 0, or invalid
            if value is None or value <= 0:
                input_data[col] = pollutant_defaults[col]
            else:
                input_data[col] = value
        
        # Add weather data
        for key, value in weather_data.items():
            input_data[f'weather_{key}'] = value
        
        # Add time features
        date_obj = pd.to_datetime(date)
        input_data['year'] = date_obj.year
        input_data['month'] = date_obj.month
        input_data['day_of_year'] = date_obj.dayofyear
        season = ((date_obj.month%12 + 3)//3)
        season_map = {1: 0, 2: 1, 3: 2, 4: 3}  # Winter, Spring, Summer, Fall
        input_data['season_encoded'] = season_map[season]
        
        # Encode city (you might need to update this based on your training data)
        city_mapping = {'Ahmedabad': 0, 'Delhi': 1, 'Mumbai': 2, 'Bengaluru': 3}  # Extend as needed
        input_data['city_encoded'] = city_mapping.get(city, 0)
        
        # Create DataFrame with all required features
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select features in the same order as training
        input_df = input_df[self.feature_columns].fillna(0)
        
        # Make prediction
        prediction = self.model.predict(input_df)[0]
        
        return prediction

def main():
    """Main function to run the complete pipeline"""
    predictor = AQIWeatherPredictor()
    
    # Step 1: Load and clean data
    print("=== Step 1: Loading and cleaning data ===")
    df_clean = predictor.load_and_clean_data()
    
    # Step 2: Create weather-enhanced dataset
    print("\n=== Step 2: Creating weather-enhanced dataset ===")
    print("This may take a while as we fetch weather data from Open Meteo API...")
    merged_df = predictor.create_weather_enhanced_dataset(df_clean)
    
    # Step 3: Train model
    print("\n=== Step 3: Training AQI prediction model ===")
    r2, mae, feature_importance = predictor.train_model(merged_df)
    
    print(f"\n=== Pipeline Complete ===")
    print(f"Final model performance - R²: {r2:.4f}, MAE: {mae:.2f}")

if __name__ == "__main__":
    main()