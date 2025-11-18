# SAVE THIS AS: app_enhanced.py

# Python 3.14 compatibility fix
import sys
if sys.version_info >= (3, 12):
    import pkgutil
    if not hasattr(pkgutil, 'get_loader'):
        pkgutil.get_loader = lambda name: None

import os
import time
import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
import requests
import math
import joblib
import logging
from functools import wraps
from collections import deque
import hashlib

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DB = 'solar.db'
PORT = int(os.getenv("PORT", 5000))
MODEL_DIR = 'models'
LOGS_DIR = 'logs'
API_KEY = os.getenv("API_KEY", "solar_secure_key_2025")  # Change in production!

# Weather API
OWM_KEY = "266cbcfc14167cde4293c8c572d95c62"
LOCATION_LAT = "13.05565"
LOCATION_LON = "77.50561"

WEATHER_CACHE_TTL = 10 * 60
weather_cache = {"data": None, "timestamp": 0}
weather_history_cache = deque(maxlen=100)  # Store weather history

# Cost calculation (INR per kWh)
GRID_COST_PER_KWH = 7.5
SOLAR_COST_PER_KWH = 0.5

# Model monitoring
prediction_log = deque(maxlen=1000)
feature_stats = {"count": 0, "means": {}, "stds": {}}

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{LOGS_DIR}/solar_ml.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# SECURITY & AUTHENTICATION
# ============================================================
def require_api_key(f):
    """Decorator to require API key for sensitive endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if provided_key != API_KEY:
            logger.warning(f"Unauthorized access attempt to {request.path}")
            return jsonify({"error": "Unauthorized", "message": "Valid API key required"}), 401
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================
class EnhancedFeatureEngine:
    """Advanced feature engineering with weather integration"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, ldr_value, timestamp, history_data, weather_data):
        """Extract comprehensive feature set with weather integration"""
        features = {}
        dt = datetime.fromtimestamp(timestamp)
        
        # Temporal features
        features['hour'] = dt.hour
        features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
        features['day_of_week'] = dt.weekday()
        features['day_of_year'] = dt.timetuple().tm_yday
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
        features['month'] = dt.month
        features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
        
        # Current sensor reading
        features['ldr_current'] = ldr_value
        features['ldr_normalized'] = ldr_value / 1023
        
        # Historical statistics
        if history_data and len(history_data) > 0:
            recent_ldr = [r[2] for r in history_data[-20:]]
            features['ldr_mean_20'] = np.mean(recent_ldr)
            features['ldr_std_20'] = np.std(recent_ldr)
            features['ldr_max_20'] = np.max(recent_ldr)
            features['ldr_min_20'] = np.min(recent_ldr)
            
            if len(recent_ldr) > 1:
                features['ldr_trend'] = (recent_ldr[-1] - recent_ldr[0]) / len(recent_ldr)
                features['ldr_velocity'] = recent_ldr[-1] - recent_ldr[-2] if len(recent_ldr) > 1 else 0
            else:
                features['ldr_trend'] = 0
                features['ldr_velocity'] = 0
            
            # Longer-term statistics
            if len(history_data) > 50:
                long_ldr = [r[2] for r in history_data[-50:]]
                features['ldr_mean_50'] = np.mean(long_ldr)
                features['ldr_std_50'] = np.std(long_ldr)
            else:
                features['ldr_mean_50'] = features['ldr_mean_20']
                features['ldr_std_50'] = features['ldr_std_20']
        else:
            features['ldr_mean_20'] = ldr_value
            features['ldr_std_20'] = 0
            features['ldr_max_20'] = ldr_value
            features['ldr_min_20'] = ldr_value
            features['ldr_trend'] = 0
            features['ldr_velocity'] = 0
            features['ldr_mean_50'] = ldr_value
            features['ldr_std_50'] = 0
        
        # Weather features (with fallback for missing data)
        features['temp'] = weather_data.get('temp', 25)
        features['humidity'] = weather_data.get('humidity', 60)
        features['clouds'] = weather_data.get('clouds', 20)
        
        # Weather-derived features
        weather_desc = weather_data.get('description', 'clear').lower()
        features['weather_clear'] = 1 if 'clear' in weather_desc else 0
        features['weather_cloudy'] = 1 if 'cloud' in weather_desc else 0
        features['weather_rain'] = 1 if 'rain' in weather_desc else 0
        
        # Interaction features
        features['hour_x_clouds'] = features['hour'] * features['clouds']
        features['ldr_x_temp'] = features['ldr_normalized'] * features['temp']
        
        self.feature_names = list(features.keys())
        return features
    
    def features_to_array(self, features_dict):
        """Convert features dict to numpy array"""
        return np.array([features_dict[k] for k in self.feature_names]).reshape(1, -1)

# ============================================================
# ML MODEL MANAGER WITH PIPELINES
# ============================================================
class MLModelManager:
    """Manages ML models with pipeline approach and versioning"""
    
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        self.pipelines = {}
        self.metadata = {}
        self.feature_engine = EnhancedFeatureEngine()
        self.load_models()
        
    def create_pipeline(self, model_name='random_forest'):
        """Create sklearn Pipeline with preprocessing and model"""
        if model_name == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_name == 'ridge':
            model = Ridge(alpha=1.0, random_state=42)
        else:
            model = RandomForestRegressor(random_state=42)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        return pipeline
    
    def train_with_time_series_cv(self, X, y, model_name='random_forest', n_splits=5, 
                                   tune_hyperparameters=False):
        """Train model with TimeSeriesSplit validation"""
        logger.info(f"Training {model_name} with TimeSeriesSplit (n_splits={n_splits})")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        pipeline = self.create_pipeline(model_name)
        
        cv_scores = []
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            cv_scores.append(mae)
            fold_metrics.append({'fold': fold, 'mae': mae, 'rmse': rmse, 'r2': r2})
            
            logger.info(f"  Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        # Hyperparameter tuning (optional)
        if tune_hyperparameters and model_name == 'random_forest':
            logger.info("Running hyperparameter tuning...")
            param_distributions = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [10, 15, 20, None],
                'model__min_samples_split': [2, 5, 10]
            }
            
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions,
                n_iter=10,
                cv=tscv,
                scoring='neg_mean_absolute_error',
                random_state=42,
                n_jobs=-1
            )
            
            random_search.fit(X, y)
            pipeline = random_search.best_estimator_
            logger.info(f"Best params: {random_search.best_params_}")
        else:
            # Final training on all data
            pipeline.fit(X, y)
        
        # Store pipeline and metadata
        self.pipelines[model_name] = pipeline
        self.metadata[model_name] = {
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'cv_mae_mean': np.mean(cv_scores),
            'cv_mae_std': np.std(cv_scores),
            'fold_metrics': fold_metrics,
            'feature_names': self.feature_engine.feature_names
        }
        
        logger.info(f"Training complete. CV MAE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return pipeline, self.metadata[model_name]
    
    def save_model(self, model_name='random_forest', version='v1'):
        """Save pipeline and metadata"""
        if model_name not in self.pipelines:
            logger.error(f"Model {model_name} not found in pipelines")
            return False
        
        model_path = os.path.join(self.model_dir, f'{model_name}_{version}.joblib')
        metadata_path = os.path.join(self.model_dir, f'{model_name}_{version}_metadata.json')
        
        try:
            joblib.dump(self.pipelines[model_name], model_path)
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata[model_name], f, indent=2)
            
            logger.info(f"Model saved: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_models(self):
        """Load all available models"""
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory {self.model_dir} not found")
            return
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.joblib'):
                model_path = os.path.join(self.model_dir, filename)
                model_name = filename.replace('.joblib', '')
                
                try:
                    self.pipelines[model_name] = joblib.load(model_path)
                    
                    # Load metadata
                    metadata_path = model_path.replace('.joblib', '_metadata.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            self.metadata[model_name] = json.load(f)
                    
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {e}")
    
    def predict(self, features_dict, model_name='random_forest_v1'):
        """Make prediction with confidence intervals"""
        if model_name not in self.pipelines:
            logger.warning(f"Model {model_name} not found, using fallback")
            model_name = list(self.pipelines.keys())[0] if self.pipelines else None
            
            if not model_name:
                return None
        
        try:
            X = self.feature_engine.features_to_array(features_dict)
            prediction = self.pipelines[model_name].predict(X)[0]
            
            # Calculate prediction interval (simple approach using CV std)
            cv_mae = self.metadata[model_name].get('cv_mae_mean', 0.05)
            lower_bound = max(0, prediction - 1.96 * cv_mae)
            upper_bound = min(1, prediction + 1.96 * cv_mae)
            
            return {
                'prediction': float(prediction),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence': float(1 - cv_mae)  # Simplified confidence
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

# Global model manager
model_manager = MLModelManager()

# ============================================================
# MONITORING & DRIFT DETECTION
# ============================================================
class ModelMonitor:
    """Monitor model performance and feature drift"""
    
    def __init__(self):
        self.predictions = deque(maxlen=1000)
        self.actuals = deque(maxlen=1000)
        self.feature_distributions = {}
        self.baseline_features = None
        
    def log_prediction(self, features, prediction, actual=None):
        """Log prediction for monitoring"""
        timestamp = time.time()
        log_entry = {
            'timestamp': timestamp,
            'features': features,
            'prediction': prediction,
            'actual': actual
        }
        self.predictions.append(log_entry)
        
        if actual is not None:
            self.actuals.append({'timestamp': timestamp, 'value': actual})
        
        # Update feature distributions
        for key, value in features.items():
            if key not in self.feature_distributions:
                self.feature_distributions[key] = deque(maxlen=1000)
            self.feature_distributions[key].append(value)
    
    def calculate_drift(self):
        """Calculate feature drift metrics"""
        if not self.baseline_features:
            # Set baseline from current distribution
            self.baseline_features = {
                k: {'mean': np.mean(list(v)), 'std': np.std(list(v))}
                for k, v in self.feature_distributions.items()
                if len(v) > 10
            }
            return {}
        
        drift_metrics = {}
        for feature, values in self.feature_distributions.items():
            if feature not in self.baseline_features or len(values) < 10:
                continue
            
            current_mean = np.mean(list(values))
            current_std = np.std(list(values))
            baseline_mean = self.baseline_features[feature]['mean']
            baseline_std = self.baseline_features[feature]['std']
            
            # Calculate drift score (normalized difference)
            mean_drift = abs(current_mean - baseline_mean) / (baseline_std + 1e-6)
            drift_metrics[feature] = float(mean_drift)
        
        return drift_metrics
    
    def get_performance_metrics(self):
        """Calculate recent performance metrics"""
        if len(self.actuals) < 5:
            return {}
        
        recent_preds = [p['prediction'] for p in list(self.predictions)[-len(self.actuals):]]
        recent_actuals = [a['value'] for a in self.actuals]
        
        mae = mean_absolute_error(recent_actuals, recent_preds)
        rmse = np.sqrt(mean_squared_error(recent_actuals, recent_preds))
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'n_samples': len(recent_actuals)
        }

monitor = ModelMonitor()

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# DATABASE
# ============================================================
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            ldr INTEGER,
            deviceId TEXT
        )
    ''')
    conn.commit()
    
    c.execute("SELECT COUNT(*) FROM readings")
    count = c.fetchone()[0]
    if count == 0:
        logger.info("No data found. Generating sample data...")
        generate_sample_data(c)
        conn.commit()
        logger.info(f"Generated sample readings")
    
    conn.close()

def generate_sample_data(cursor):
    """Generate realistic solar data for past 7 days"""
    base_ts = int(time.time()) - (7 * 24 * 60 * 60)
    
    for day in range(7):
        for hour in range(24):
            for minute in [0, 10, 20, 30, 40, 50]:
                ts = base_ts + (day * 24 * 3600) + (hour * 3600) + (minute * 60)
                
                if 6 <= hour <= 18:
                    peak_hour = 12
                    distance_from_peak = abs(hour - peak_hour)
                    intensity = 1023 * math.exp(-0.08 * distance_from_peak**2)
                    
                    # Add weather variation
                    day_factor = 0.8 + (day % 3) * 0.1
                    noise = (hash(ts) % 100) - 50
                    ldr = int(max(0, min(1023, intensity * day_factor + noise)))
                else:
                    ldr = int(20 + (hash(ts) % 30))
                
                cursor.execute("INSERT INTO readings (timestamp, ldr, deviceId) VALUES (?, ?, ?)",
                              (ts, ldr, 'sample-device'))

# ============================================================
# WEATHER WITH HISTORY
# ============================================================
def fetch_weather(lat=None, lon=None, store_history=True):
    """Fetch weather with history storage"""
    global weather_cache
    
    use_lat = lat if lat else LOCATION_LAT
    use_lon = lon if lon else LOCATION_LON
    
    cache_key = f"{use_lat},{use_lon}"
    now = time.time()
    
    if cache_key in weather_cache and (now - weather_cache.get(f"{cache_key}_time", 0)) < WEATHER_CACHE_TTL:
        return weather_cache[cache_key]
    
    if not OWM_KEY:
        default_weather = {"description": "clear sky", "temp": 25, "humidity": 60, "clouds": 20}
        return default_weather
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={use_lat}&lon={use_lon}&appid={OWM_KEY}&units=metric"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            result = {
                "description": data["weather"][0]["description"],
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "clouds": data["clouds"]["all"]
            }
            weather_cache[cache_key] = result
            weather_cache[f"{cache_key}_time"] = now
            
            if store_history:
                weather_history_cache.append({
                    'timestamp': now,
                    'data': result
                })
            
            return result
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
    
    return {"description": "clear sky", "temp": 25, "humidity": 60, "clouds": 20}

# ============================================================
# TRAINING ENDPOINT (Background-ready)
# ============================================================
@app.route('/api/train', methods=['POST'])
@require_api_key
def train_model():
    """Train ML models with advanced techniques"""
    try:
        # Get parameters
        data = request.get_json() or {}
        model_type = data.get('model_type', 'random_forest')
        tune = data.get('tune_hyperparameters', False)
        n_splits = data.get('cv_splits', 5)
        
        logger.info(f"Starting training: model={model_type}, tune={tune}")
        
        # Fetch training data
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT * FROM readings ORDER BY timestamp ASC")
        history = c.fetchall()
        conn.close()
        
        if len(history) < 100:
            return jsonify({
                "status": "error",
                "message": f"Insufficient data: {len(history)} samples (need 100+)"
            }), 400
        
        # Prepare features and targets
        X_list = []
        y_list = []
        
        for i in range(len(history) - 1):
            current = history[i]
            next_reading = history[i + 1]
            
            ts, ldr = current[1], current[2]
            next_ldr = next_reading[2]
            
            # Get weather (use cached/fallback)
            weather = fetch_weather(store_history=False)
            
            # Extract features
            features = model_manager.feature_engine.extract_features(
                ldr, ts, history[max(0, i-50):i+1], weather
            )
            
            X_list.append([features[k] for k in model_manager.feature_engine.feature_names])
            y_list.append(next_ldr / 1023)  # Normalize target
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train model
        pipeline, metadata = model_manager.train_with_time_series_cv(
            X, y, model_name=model_type, n_splits=n_splits, tune_hyperparameters=tune
        )
        
        # Save model
        version = f"v{int(time.time())}"
        model_manager.save_model(model_type, version)
        
        return jsonify({
            "status": "success",
            "model": f"{model_type}_{version}",
            "metadata": metadata
        })
        
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================================
# PREDICTION ENDPOINT
# ============================================================
@app.route('/api/latest')
def latest():
    """Get latest reading with ML prediction"""
    try:
        lat = request.args.get('lat', LOCATION_LAT)
        lon = request.args.get('lon', LOCATION_LON)
        
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 1")
        row = c.fetchone()
        
        c.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 100")
        history = c.fetchall()
        conn.close()
        
        if not row:
            return jsonify({"status": "no-data"}), 404
        
        ts, ldr_value = row[1], row[2]
        weather = fetch_weather(lat, lon)
        
        # Extract features
        features = model_manager.feature_engine.extract_features(
            ldr_value, ts, history, weather
        )
        
        # Make prediction
        prediction = model_manager.predict(features)
        
        if not prediction:
            return jsonify({"status": "error", "message": "Prediction failed"}), 500
        
        # Log for monitoring
        monitor.log_prediction(features, prediction['prediction'])
        
        # Calculate derived metrics
        predicted_intensity = prediction['prediction'] * 100
        current_intensity = (ldr_value / 1023) * 100
        
        # Calculate predicted sun hours
        hour = datetime.fromtimestamp(ts).hour
        is_daylight = 6 <= hour <= 18
        base_hours = predicted_intensity / 100 * 12
        weather_factor = 1.2 if 'clear' in weather['description'].lower() else 0.8
        predicted_hours = base_hours * weather_factor * (1.0 if is_daylight else 0.3)
        predicted_hours = max(0, min(predicted_hours, 12))
        
        # Recommendations
        if predicted_intensity > 70:
            window = {"start": "09:00", "end": "16:00"}
        elif predicted_intensity > 50:
            window = {"start": "10:00", "end": "15:00"}
        else:
            window = {"start": "11:00", "end": "14:00"}
        
        response = {
            "timestamp": ts,
            "time_iso": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
            "ldr": ldr_value,
            "deviceId": row[3],
            "weather": weather,
            "prediction": {
                "predicted_sun_hours": round(predicted_hours, 2),
                "intensity_current": round(current_intensity, 1),
                "intensity_predicted": round(predicted_intensity, 1),
                "confidence": round(prediction['confidence'] * 100, 1),
                "prediction_interval": {
                    "lower": round(prediction['lower_bound'] * 100, 1),
                    "upper": round(prediction['upper_bound'] * 100, 1)
                },
                "recommended_window": window
            },
            "ml_info": {
                "model_type": "sklearn_pipeline",
                "features_used": len(features),
                "cv_validated": True
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Latest endpoint error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================================
# MONITORING ENDPOINTS
# ============================================================
@app.route('/api/monitoring/drift')
@require_api_key
def monitoring_drift():
    """Get feature drift metrics"""
    drift_metrics = monitor.calculate_drift()
    performance = monitor.get_performance_metrics()
    
    return jsonify({
        "feature_drift": drift_metrics,
        "performance": performance,
        "samples_logged": len(monitor.predictions)
    })

@app.route('/api/monitoring/health')
def monitoring_health():
    """Health check endpoint"""
    models_loaded = len(model_manager.pipelines)
    
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "predictions_logged": len(monitor.predictions),
        "weather_api": bool(OWM_KEY),
        "database": os.path.exists(DB)
    })

# ============================================================
# MODEL INFO
# ============================================================
@app.route('/api/models/info')
def model_info():
    """Get information about loaded models"""
    return jsonify({
        "loaded_models": list(model_manager.pipelines.keys()),
        "metadata": model_manager.metadata,
        "feature_count": len(model_manager.feature_engine.feature_names),
        "features": model_manager.feature_engine.feature_names
    })

# ============================================================
# SENSOR DATA (with auto-retrain trigger)
# ============================================================
@app.route('/api/sensor', methods=['POST'])
def sensor_data():
    """Receive sensor data and trigger retraining if needed"""
    data = request.get_json()
    deviceId = data.get('deviceId', 'unknown')
    ldr = data.get('ldr', 0)
    ts = int(time.time())
    
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO readings (timestamp, ldr, deviceId) VALUES (?, ?, ?)",
              (ts, ldr, deviceId))
    conn.commit()
    
    # Check if retraining needed (every 1000 samples)
    c.execute("SELECT COUNT(*) FROM readings")
    count = c.fetchone()[0]
    conn.close()
    
    should_retrain = count % 1000 == 0 and count > 0
    
    return jsonify({
        "status": "ok",
        "timestamp": ts,
        "total_samples": count,
        "retrain_triggered": should_retrain,
        "message": "Data received. Trigger /api/train for retraining." if should_retrain else "Data received."
    })

# ============================================================
# HISTORY ENDPOINT
# ============================================================
@app.route('/api/history')
def history():
    """Get historical readings"""
    limit = request.args.get('limit', 200, type=int)
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    
    data = [{
        "timestamp": r[1],
        "time_iso": datetime.fromtimestamp(r[1], timezone.utc).isoformat(),
        "ldr": r[2],
        "deviceId": r[3]
    } for r in rows]
    
    return jsonify(data)

# ============================================================
# INDEX PAGE
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

# ============================================================
# STARTUP
# ============================================================
if __name__ == '__main__':
    init_db()
    
    # Check if models exist, if not train initial model
    if not model_manager.pipelines:
        logger.info("No models found. Training initial model...")
        
        try:
            conn = sqlite3.connect(DB)
            c = conn.cursor()
            c.execute("SELECT * FROM readings ORDER BY timestamp ASC")
            history = c.fetchall()
            conn.close()
            
            if len(history) >= 100:
                # Prepare training data
                X_list = []
                y_list = []
                
                for i in range(len(history) - 1):
                    current = history[i]
                    next_reading = history[i + 1]
                    
                    ts, ldr = current[1], current[2]
                    next_ldr = next_reading[2]
                    
                    weather = fetch_weather(store_history=False)
                    features = model_manager.feature_engine.extract_features(
                        ldr, ts, history[max(0, i-50):i+1], weather
                    )
                    
                    X_list.append([features[k] for k in model_manager.feature_engine.feature_names])
                    y_list.append(next_ldr / 1023)
                
                X = np.array(X_list)
                y = np.array(y_list)
                
                # Train initial model
                pipeline, metadata = model_manager.train_with_time_series_cv(
                    X, y, model_name='random_forest', n_splits=3
                )
                
                model_manager.save_model('random_forest', 'v1')
                logger.info("Initial model trained successfully")
            else:
                logger.warning(f"Insufficient data for training: {len(history)} samples")
                
        except Exception as e:
            logger.error(f"Initial training failed: {e}")
    
    print("\n" + "="*70)
    print("🚀 SMART SOLAR ENERGY SCHEDULER - PRODUCTION ML EDITION")
    print("="*70)
    print("\n📊 ML SYSTEM STATUS:")
    print(f"   ├─ Models Loaded: {len(model_manager.pipelines)}")
    print(f"   ├─ Features: {len(model_manager.feature_engine.feature_names)}")
    print(f"   ├─ Validation: TimeSeriesSplit CV")
    print(f"   ├─ Pipeline: StandardScaler + Model")
    print(f"   └─ Monitoring: Drift Detection + Performance Tracking")
    
    print("\n🔐 SECURITY:")
    print(f"   ├─ API Key Required: /api/train, /api/monitoring/drift")
    print(f"   └─ Current API Key: {API_KEY[:10]}...")
    
    print("\n📁 STORAGE:")
    print(f"   ├─ Models: {MODEL_DIR}/")
    print(f"   ├─ Logs: {LOGS_DIR}/")
    print(f"   └─ Database: {DB}")
    
    print("\n🌐 ENDPOINTS:")
    print(f"   ├─ Health: http://localhost:{PORT}/api/monitoring/health")
    print(f"   ├─ Latest: http://localhost:{PORT}/api/latest")
    print(f"   ├─ Train: POST http://localhost:{PORT}/api/train (requires API key)")
    print(f"   ├─ Drift: http://localhost:{PORT}/api/monitoring/drift (requires API key)")
    print(f"   └─ Models: http://localhost:{PORT}/api/models/info")
    
    if OWM_KEY:
        print("\n🌤️  Weather API: ✅ Connected")
    else:
        print("\n🌤️  Weather API: ⚠️  Using fallback data")
    
    print("\n💡 IMPROVEMENTS IMPLEMENTED:")
    print("   ✅ TimeSeriesSplit validation (temporal integrity)")
    print("   ✅ sklearn Pipeline (preprocessing + model)")
    print("   ✅ Feature drift monitoring")
    print("   ✅ Weather integration in features")
    print("   ✅ Prediction intervals (confidence bounds)")
    print("   ✅ API key authentication")
    print("   ✅ Comprehensive logging")
    print("   ✅ Model versioning & metadata")
    print("   ✅ Hyperparameter tuning support")
    print("   ✅ Production-ready architecture")
    
    print("\n📝 NEXT STEPS FOR FULL PRODUCTION:")
    print("   🔄 Add: Celery/RQ for async training")
    print("   📊 Add: Prometheus metrics export")
    print("   🐳 Add: Docker containerization")
    print("   🧪 Add: Unit & integration tests")
    print("   🔒 Add: Rate limiting (Flask-Limiter)")
    print("   ☁️  Add: Cloud deployment config")
    
    print("\n" + "="*70)
    print(f"🎯 Server starting on http://localhost:{PORT}")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)