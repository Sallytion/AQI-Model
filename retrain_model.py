import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def retrain_existing_model():
    """Retrain the model to predict individual pollutant values instead of combined AQI"""
    
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
    
    print("Preparing features for multi-output pollutant prediction model...")
    
    # Define target pollutants to predict
    target_pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    print(f"Target pollutants to predict: {target_pollutants}")
    
    # Prepare features (exclude target pollutants from features)
    feature_columns = []
    
    # Weather features (these will be our main predictors)
    weather_cols = [col for col in merged_df.columns if col.startswith('weather_')]
    feature_columns.extend(weather_cols)
    print(f"Weather features ({len(weather_cols)}): {weather_cols}")
    
    # Add other pollutants as features (but not the ones we're predicting)
    other_pollutants = ['NO', 'NOx', 'NH3', 'Benzene', 'Toluene', 'Xylene']
    for pol in other_pollutants:
        if pol in merged_df.columns:
            feature_columns.append(pol)
    
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
    available_targets = [col for col in target_pollutants if col in merged_df.columns]
    
    print(f"Available features ({len(available_features)}): {available_features}")
    print(f"Available targets ({len(available_targets)}): {available_targets}")
    
    X = merged_df[available_features].copy()
    y = merged_df[available_targets].copy()
    
    # Handle missing values more robustly
    print("\nChecking data quality...")
    print(f"Features missing data per column:\n{X.isnull().sum()}")
    print(f"Targets missing data per column:\n{y.isnull().sum()}")
    
    # For features: fill with 0 (safer for weather data and pollutants)
    X = X.fillna(0)
    
    # For targets: fill with column means, but use 0 if all values are NaN
    for col in y.columns:
        if y[col].isnull().all():
            print(f"Warning: All values in {col} are NaN. Filling with default values.")
            # Set reasonable defaults for each pollutant
            defaults = {'PM2.5': 50, 'PM10': 80, 'NO2': 40, 'SO2': 30, 'CO': 1.5, 'O3': 80}
            y[col] = defaults.get(col, 50)
        else:
            y[col] = y[col].fillna(y[col].mean())
    
    # Fill missing values with median for numeric columns
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(0)
    
    # Fill missing values in target variables
    for col in y.columns:
        y[col] = y[col].fillna(y[col].median())
    
    print(f"Feature data shape: {X.shape}")
    print(f"Target data shape: {y.shape}")
    
    # Final verification - no NaN values should remain
    assert not X.isnull().any().any(), f"Still have NaN in features: {X.isnull().sum().sum()}"
    assert not y.isnull().any().any(), f"Still have NaN in targets: {y.isnull().sum().sum()}"
    print("✅ Data cleaning complete - no NaN values remaining")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training Multi-Output Random Forest model for individual pollutant prediction...")
    # Create multi-output model
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    print(f"\nMulti-Output Model Performance:")
    print("=" * 50)
    
    # Calculate metrics for each pollutant
    pollutant_metrics = {}
    for i, pollutant in enumerate(available_targets):
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        pollutant_metrics[pollutant] = {'r2': r2, 'mae': mae}
        print(f"{pollutant:>6}: R² = {r2:.4f}, MAE = {mae:.2f}")
    
    # Overall metrics
    overall_r2 = np.mean([metrics['r2'] for metrics in pollutant_metrics.values()])
    overall_mae = np.mean([metrics['mae'] for metrics in pollutant_metrics.values()])
    print(f"{'Overall':>6}: R² = {overall_r2:.4f}, MAE = {overall_mae:.2f}")
    
    # Feature importance for each pollutant
    print("\nFeature Importance Analysis:")
    print("=" * 50)
    
    feature_importance_df = pd.DataFrame(index=available_features)
    for i, pollutant in enumerate(available_targets):
        importance = model.estimators_[i].feature_importances_
        feature_importance_df[f'{pollutant}_importance'] = importance
    
    # Average importance across all pollutants
    feature_importance_df['avg_importance'] = feature_importance_df.mean(axis=1)
    feature_importance_df = feature_importance_df.sort_values('avg_importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance_df[['avg_importance']].head(10))
    
    print("\nTop 10 Feature Importance:")
    print(feature_importance_df[['avg_importance']].head(10))
    
    # Save model and features 
    print("\nSaving multi-output pollutant prediction model...")
    joblib.dump(model, 'aqi_weather_model.joblib')
    joblib.dump(available_features, 'aqi_weather_features.joblib')
    joblib.dump(available_targets, 'aqi_weather_targets.joblib')  # Save target pollutants
    joblib.dump(pollutant_metrics, 'aqi_weather_metrics.joblib')  # Save metrics
    
    print("✅ Multi-Output Pollutant Prediction Model trained and saved successfully!")
    print(f"Model Type: Multi-Output Random Forest")
    print(f"Predicts: {available_targets}")
    print(f"Using Features: {len(available_features)} features")
    print(f"Overall Performance - R²: {overall_r2:.4f}, MAE: {overall_mae:.2f}")
    print("\nThis model predicts individual pollutant concentrations instead of combined AQI.")
    print("AQI can be calculated from individual pollutant predictions using standard formulas.")

def calculate_aqi_from_pollutants(pollutant_dict):
    """
    Calculate AQI from individual pollutant concentrations using standard AQI formulas.
    This is more accurate than predicting AQI directly.
    """
    # Standard AQI breakpoints for India (CPCB)
    aqi_breakpoints = {
        'PM2.5': [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200), 
                  (90, 120, 201, 300), (120, 250, 301, 400), (250, float('inf'), 401, 500)],
        'PM10': [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
                 (250, 350, 201, 300), (350, 430, 301, 400), (430, float('inf'), 401, 500)],
        'NO2': [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200),
                (180, 280, 201, 300), (280, 400, 301, 400), (400, float('inf'), 401, 500)],
        'SO2': [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200),
                (380, 800, 201, 300), (800, 1600, 301, 400), (1600, float('inf'), 401, 500)],
        'CO': [(0, 1.0, 0, 50), (1.0, 2.0, 51, 100), (2.0, 10, 101, 200),
               (10, 17, 201, 300), (17, 34, 301, 400), (34, float('inf'), 401, 500)],
        'O3': [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200),
               (168, 208, 201, 300), (208, 748, 301, 400), (748, float('inf'), 401, 500)]
    }
    
    def get_aqi_for_pollutant(pollutant, concentration):
        if pollutant not in aqi_breakpoints:
            return 0
        
        for bp_lo, bp_hi, aqi_lo, aqi_hi in aqi_breakpoints[pollutant]:
            if bp_lo <= concentration <= bp_hi:
                # Linear interpolation
                aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + aqi_lo
                return int(aqi)
        return 500  # Maximum AQI
    
    # Calculate AQI for each pollutant
    aqi_values = []
    for pollutant, concentration in pollutant_dict.items():
        if pollutant in aqi_breakpoints and concentration > 0:
            aqi_val = get_aqi_for_pollutant(pollutant, concentration)
            aqi_values.append(aqi_val)
    
    # Return maximum AQI (worst pollutant determines overall AQI)
    return max(aqi_values) if aqi_values else 50

if __name__ == "__main__":
    retrain_existing_model()