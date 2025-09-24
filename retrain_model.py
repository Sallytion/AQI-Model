import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def retrain_existing_model():
    """Retrain the model using existing merged data to fix scikit-learn version compatibility"""
    
    print("Loading existing merged dataset...")
    try:
        # Load the existing merged dataset
        merged_df = pd.read_csv('aqi_weather_merged.csv')
        print(f"Loaded merged dataset with shape: {merged_df.shape}")
    except FileNotFoundError:
        print("Error: aqi_weather_merged.csv not found!")
        print("Please make sure the weather data processing was completed successfully.")
        return
    
    # Convert Date column to datetime
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    
    print("Preparing features for model training...")
    
    # Prepare features (same as in original script)
    feature_columns = []
    
    # Pollutant features
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    feature_columns.extend(pollutant_cols)
    
    # Weather features
    weather_cols = [col for col in merged_df.columns if col.startswith('weather_')]
    feature_columns.extend(weather_cols)
    
    # Time features
    merged_df['year'] = merged_df['Date'].dt.year
    merged_df['month'] = merged_df['Date'].dt.month
    merged_df['day_of_year'] = merged_df['Date'].dt.dayofyear
    merged_df['season'] = ((merged_df['Date'].dt.month%12 + 3)//3).map({1:'Winter', 2:'Spring', 3:'Summer', 4:'Fall'})
    
    # Encode categorical features
    merged_df['season_encoded'] = pd.Categorical(merged_df['season']).codes
    merged_df['city_encoded'] = pd.Categorical(merged_df['City']).codes
    
    time_features = ['year', 'month', 'day_of_year', 'season_encoded', 'city_encoded']
    feature_columns.extend(time_features)
    
    # Select only available columns
    available_features = [col for col in feature_columns if col in merged_df.columns]
    
    X = merged_df[available_features].copy()
    y = merged_df['AQI'].copy()
    
    # Fill missing values with median for numeric columns
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(0)
    
    print(f"Feature columns ({len(available_features)}): {available_features}")
    print(f"Training data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print("Training Random Forest model with current scikit-learn version...")
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importance:")
    print(feature_importance.head(10))
    
    # Save model and features (this will overwrite the old ones)
    print("\nSaving updated model...")
    joblib.dump(model, 'aqi_weather_model.joblib')
    joblib.dump(available_features, 'aqi_weather_features.joblib')
    
    print("✅ Model retrained and saved successfully!")
    print(f"Final model performance - R²: {r2:.4f}, MAE: {mae:.2f}")
    print("The model is now compatible with the current scikit-learn version.")

if __name__ == "__main__":
    retrain_existing_model()