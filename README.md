# Air Quality Index (AQI) Dashboard & Prediction System

A comprehensive Streamlit-based dashboard for analyzing, visualizing, and predicting Air Quality Index (AQI) data across India. This application combines historical data analysis with real-time monitoring and machine learning predictions.

## 🌟 Overview

This project provides a complete solution for air quality monitoring and analysis, featuring:
- Historical AQI data analysis and visualization
- Real-time air quality monitoring using government APIs
- Machine learning-based AQI prediction
- Interactive maps and charts
- Comprehensive statistical insights

## 🚀 Technologies Used

### Frontend & UI Framework
- **Streamlit** - Web application framework for rapid prototyping and deployment
  - *Why chosen*: Provides rapid development of data-driven web applications with minimal code
  - *Use case*: Dashboard interface, interactive widgets, and data visualization display

### Data Processing & Analysis
- **Pandas** - Data manipulation and analysis library
  - *Why chosen*: Excellent for handling CSV data, time series analysis, and data cleaning
  - *Use case*: Loading CSV files, data preprocessing, feature engineering, and statistical analysis

- **NumPy** - Numerical computing library
  - *Why chosen*: Efficient array operations and mathematical computations
  - *Use case*: Mathematical operations, array manipulations, and statistical calculations

### Machine Learning
- **Scikit-learn** - Machine learning library
  - *Why chosen*: Comprehensive ML algorithms with excellent documentation and performance
  - *Use case*: Model training, evaluation, and prediction
  - **Models implemented**:
    - **Random Forest Regressor** - For AQI value prediction
    - **Random Forest Classifier** - For AQI category classification

- **Joblib** - Model persistence library
  - *Why chosen*: Efficient serialization of large NumPy arrays in scikit-learn models
  - *Use case*: Saving and loading trained models for reuse

### Data Visualization
- **Plotly** - Interactive plotting library
  - *Why chosen*: Creates interactive, publication-quality graphs and maps
  - *Use case*: Interactive charts, time series plots, and geographical mapping
  - **Components used**:
    - `plotly.express` - High-level interface for quick visualizations
    - `plotly.graph_objects` - Low-level interface for custom visualizations
    - `scatter_mapbox` - Interactive geographical mapping

### External Data Integration
- **Requests** - HTTP library for API calls
  - *Why chosen*: Simple and elegant HTTP requests handling
  - *Use case*: Fetching real-time data from Government of India AQI API

## 🤖 Machine Learning Models

### 1. Random Forest Regressor
- **Purpose**: Predicting exact AQI numerical values
- **Algorithm**: Ensemble method using multiple decision trees
- **Why chosen**: 
  - Robust to overfitting
  - Handles non-linear relationships well
  - Provides feature importance insights
  - Works well with mixed data types
- **Features used**: Year, Month, Day, Hour, encoded Location, encoded Pollutant
- **Performance metrics**: R², MAE, MSE, RMSE

### 2. Random Forest Classifier
- **Purpose**: Classifying AQI into categories (Good, Moderate, Poor, etc.)
- **Algorithm**: Ensemble classification using multiple decision trees
- **Why chosen**:
  - Excellent for multi-class classification
  - Provides probability estimates
  - Robust to outliers
  - Interpretable results
- **Features used**: Same as regressor
- **Performance metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## 📊 Data Sources & Structure

### Historical Data Files

#### 1. Real-time Air Quality Index from Various Locations
- **File**: `Real time Air Quality Index from various locations.csv`
- **Purpose**: Provides current snapshot of AQI across monitoring stations
- **Why used**: Essential for understanding current air quality trends and patterns
- **Key columns**: Location, Pollutant, Value, Timestamp
- **Data characteristics**: Real-time measurements, multiple pollutants per location

#### 2. State/UT-wise AQI Details (2021)
- **File**: `StateUTs-wise Details of the Air Quality Index (AQI) of the Country during 2021.csv`
- **Purpose**: Annual overview of air quality across Indian states and union territories
- **Why used**: Provides state-level aggregated data for macro-level analysis
- **Key columns**: State/UT, AQI values, Pollutant concentrations
- **Data characteristics**: Administrative boundary-based aggregation, annual summary

#### 3. Delhi NCR Yearly AQI Trends (2022-2024)
- **File**: `Year-wise Details of Air Quality Index (AQI) levels in DelhiNCR from 2022 to 2024.csv`
- **Purpose**: Detailed time series data for Delhi National Capital Region
- **Why used**: Delhi NCR is a critical pollution hotspot requiring detailed analysis
- **Key columns**: Date, AQI, Pollutant levels, Categories
- **Data characteristics**: Time series data, high pollution region focus

### Real-time Data Source

#### Government of India AQI API
- **Endpoint**: Central Pollution Control Board API
- **Authentication**: Government-provided API key (unlimited access)
- **Why chosen**: 
  - Official government data source
  - Real-time updates
  - Comprehensive coverage across India
  - Reliable and authoritative
- **API Key**: 579b464db66ec23bdd0000015eb88b6f030349cb4f46c4631fb80919 (unlimited access)
- **Data frequency**: Updated every few hours
- **Coverage**: 3,000+ monitoring stations across India
- **Batch size**: Up to 1,000 records per request

## 🎯 Feature Engineering Strategy

### Temporal Features
- **Year, Month, Day, Hour**: Extracted from datetime columns
- **Reason**: Air quality shows strong temporal patterns (seasonal, daily cycles)
- **Impact**: Enables models to capture time-based pollution trends

### Categorical Encoding
- **Location Encoding**: Convert location names to numerical values
- **Pollutant Encoding**: Encode different pollutant types
- **Reason**: Machine learning algorithms require numerical inputs
- **Method**: Label encoding for maintaining ordinal relationships

## 📈 Dashboard Features

### 1. Historical Analysis Tab
- **Purpose**: Explore patterns in historical AQI data
- **Visualizations**: 
  - Time series plots
  - Distribution histograms
  - Correlation matrices
  - Statistical summaries
- **Insights**: Long-term trends, seasonal patterns, pollution hotspots

### 2. AQI Prediction Tool
- **Purpose**: Predict future AQI values and categories
- **Input**: Location, date/time, pollutant type
- **Output**: Predicted AQI value and category with confidence metrics
- **Models**: Real-time inference using trained Random Forest models

### 3. Real-time Insights
- **Purpose**: Monitor current air quality conditions
- **Data**: Live API data from government monitoring stations
- **Features**:
  - Interactive maps with station markers
  - Real-time AQI values and trends
  - Filtering by location and pollutant
  - Hover details for each monitoring station

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
Internet connection (for API access)
```

### Installation Steps
```bash
# Clone the repository
git clone <repository-url>
cd AQI-Dashboard

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Configuration
1. Ensure CSV files are in the project directory
2. Verify internet connection for API access
3. API key is embedded in the application (government-provided unlimited access)

## 🔧 Technical Implementation Details

### System Architecture
```
CSV Files → Data Loading → Data Cleaning → Feature Engineering → Model Training
    ↓
Real-time API → Data Fetching → Data Processing → Live Predictions
    ↓
Interactive Dashboard → Visualizations → User Insights
```

### Performance Optimizations
- **Caching**: Streamlit caching for data loading and model training
- **Lazy Loading**: Load data only when needed
- **Model Persistence**: Save trained models to avoid retraining
- **API Pagination**: Efficient handling of large API responses

### Error Handling
- **Data Validation**: Comprehensive checks for data integrity
- **API Resilience**: Retry mechanisms and fallback options
- **User Feedback**: Clear error messages and loading indicators
- **NaN Handling**: Robust handling of missing values in visualizations

## 📊 Model Performance

### Evaluation Metrics
- **Regression Model**: R², MAE, MSE, RMSE
- **Classification Model**: Accuracy, Precision, Recall, F1-score
- **Cross-validation**: 5-fold cross-validation for robust evaluation
- **Feature Importance**: Analysis of most predictive features

### Expected Performance
- **AQI Prediction Accuracy**: 85-90% for category classification
- **Numerical Prediction**: R² > 0.8 for continuous values
- **Real-time Processing**: < 2 seconds for API data fetching and prediction

## 🌍 Use Cases & Applications

### Environmental Monitoring
- Track air quality trends across regions
- Identify pollution hotspots and patterns
- Monitor effectiveness of pollution control measures

### Public Health
- Provide early warnings for poor air quality days
- Help citizens plan outdoor activities
- Support health advisory systems

### Policy Making
- Evidence-based environmental policy development
- Assessment of industrial impact on air quality
- Urban planning and pollution control strategies

## 🔮 Future Enhancements

### Technical Improvements
- **Enhanced Models**: Deep learning models for better accuracy
- **Real-time Predictions**: Continuous model updates with streaming data
- **Mobile App**: React Native or Flutter mobile application
- **API Development**: RESTful API for third-party integrations

### Feature Additions
- **Weather Integration**: Incorporate meteorological data
- **Health Recommendations**: Personalized health advice based on AQI
- **Alert System**: Email/SMS notifications for poor air quality
- **Comparative Analysis**: Compare multiple cities simultaneously

## 📝 Contributing

We welcome contributions! Please feel free to submit pull requests, report bugs, or suggest new features.

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Central Pollution Control Board, Government of India for API access
- Streamlit team for the excellent framework
- Scikit-learn community for machine learning tools
- Plotly team for interactive visualization capabilities

---

**Note**: This dashboard is designed for educational and research purposes. For critical decision-making, please consult official government sources and environmental agencies.
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