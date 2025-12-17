print("🔥🔥🔥 THIS IS APP3.PY RUNNING 🔥🔥🔥")
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
from collections import deque, OrderedDict
import hashlib
import secrets
from threading import Lock
import re
import tempfile

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
# CONFIGURATION - ENHANCED & SECURE
# ============================================================
# Create directories FIRST to prevent logging errors
DB = 'solar.db'
PORT = int(os.getenv("PORT", 5000))
MODEL_DIR = 'models'
LOGS_DIR = 'logs'
CACHE_DIR = 'cache'

# Create all required directories
for directory in [MODEL_DIR, LOGS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'solar_ml.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Security configuration
API_KEY = "solar2025"

print(f"\n{'='*70}")
print(f"🔑 SERVER API KEY: {API_KEY}")
print(f"{'='*70}\n")
OWM_KEY = os.getenv("OWM_KEY")  # OpenWeatherMap API key

# Location configuration (Bangalore, India)
LOCATION_LAT = os.getenv("LOCATION_LAT", "13.05565")
LOCATION_LON = os.getenv("LOCATION_LON", "77.50561")
LOCATION_TZ = os.getenv("LOCATION_TZ", "Asia/Kolkata")

# Cost calculation (INR per kWh)
GRID_COST_PER_KWH = float(os.getenv("GRID_COST_PER_KWH", "7.5"))
SOLAR_COST_PER_KWH = float(os.getenv("SOLAR_COST_PER_KWH", "0.5"))

# ============================================================
# ENHANCED WEATHER CACHE WITH PERSISTENCE
# ============================================================
class EnhancedWeatherCache:
    def __init__(self, max_size=20, ttl=600, cache_dir=CACHE_DIR):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = Lock()
        self.cache_file = os.path.join(cache_dir, 'weather_cache.json')
        self.load_persistent_cache()
    
    def load_persistent_cache(self):
        """Load cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    current_time = time.time()
                    for key, (cache_data, timestamp) in data.items():
                        if current_time - timestamp < self.ttl:
                            self.cache[key] = (cache_data, timestamp)
                logger.info(f"Loaded {len(self.cache)} weather cache entries from disk")
        except Exception as e:
            logger.warning(f"Could not load weather cache: {e}")
    
    def save_persistent_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Could not save weather cache: {e}")
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.cache.move_to_end(key)
                    return data
                else:
                    del self.cache[key]
                    self.save_persistent_cache()
            return None
    
    def set(self, key, data):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
            self.cache[key] = (data, time.time())
            self.save_persistent_cache()

# Initialize enhanced weather cache
weather_cache = EnhancedWeatherCache(max_size=20, ttl=600)

# ============================================================
# ENHANCED FEATURE ENGINEERING
# ============================================================
class EnhancedFeatureEngine:
    """Advanced feature engineering with seasonality and weather integration"""
    
    def __init__(self):
        self.feature_names = [
            'hour', 'hour_sin', 'hour_cos', 
            'day_of_week', 'day_of_year', 'month',
            'day_sin', 'day_cos', 'season',
            'is_weekend', 'is_holiday',
            'ldr_current', 'ldr_normalized',
            'ldr_mean_10', 'ldr_std_10', 'ldr_max_10', 'ldr_min_10',
            'ldr_mean_30', 'ldr_std_30', 'ldr_max_30', 'ldr_min_30',
            'ldr_trend', 'ldr_velocity', 'ldr_acceleration',
            'temp', 'humidity', 'clouds', 'pressure', 'visibility',
            'weather_clear', 'weather_cloudy', 'weather_rain', 'weather_extreme',
            'solar_noon_distance', 'daylight_remaining',
            'hour_x_clouds', 'ldr_x_temp', 'temp_x_humidity'
        ]
        
    def get_season(self, month):
        """Get season based on month (for India)"""
        if month in [12, 1, 2]:  # Winter
            return 0
        elif month in [3, 4, 5]:  # Summer
            return 1
        elif month in [6, 7, 8, 9]:  # Monsoon
            return 2
        else:  # Post-monsoon
            return 3
    
    def is_holiday(self, dt):
        """Simple holiday detection (India major holidays)"""
        # This is a simplified version - expand as needed
        holidays = [
            (1, 26),   # Republic Day
            (8, 15),   # Independence Day
            (10, 2),   # Gandhi Jayanti
        ]
        return (dt.month, dt.day) in holidays
    
    def extract_features(self, ldr_value, timestamp, history_data, weather_data):
        """Extract comprehensive feature set with enhanced temporal features"""
        features = {}
        dt = datetime.fromtimestamp(timestamp, timezone.utc)
        
        # Enhanced temporal features
        features['hour'] = dt.hour
        features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
        features['day_of_week'] = dt.weekday()
        features['day_of_year'] = dt.timetuple().tm_yday
        features['month'] = dt.month
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
        features['season'] = self.get_season(dt.month)
        features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
        features['is_holiday'] = 1 if self.is_holiday(dt) else 0
        
        # Solar position features
        features['solar_noon_distance'] = abs(dt.hour - 12)
        features['daylight_remaining'] = max(0, 18 - dt.hour) if dt.hour >= 6 else 0
        
        # Current sensor reading
        features['ldr_current'] = ldr_value
        features['ldr_normalized'] = ldr_value / 1023.0
        
        # Enhanced historical statistics
        if history_data and len(history_data) > 0:
            # Short-term statistics (last 10 readings)
            recent_ldr_10 = [r[2] for r in history_data[-10:]]
            features['ldr_mean_10'] = np.mean(recent_ldr_10)
            features['ldr_std_10'] = np.std(recent_ldr_10) if len(recent_ldr_10) > 1 else 0
            features['ldr_max_10'] = np.max(recent_ldr_10)
            features['ldr_min_10'] = np.min(recent_ldr_10)
            
            # Medium-term statistics (last 30 readings)
            recent_ldr_30 = [r[2] for r in history_data[-30:]]
            features['ldr_mean_30'] = np.mean(recent_ldr_30)
            features['ldr_std_30'] = np.std(recent_ldr_30) if len(recent_ldr_30) > 1 else 0
            features['ldr_max_30'] = np.max(recent_ldr_30)
            features['ldr_min_30'] = np.min(recent_ldr_30)
            
            # Trend analysis
            if len(recent_ldr_10) > 1:
                features['ldr_trend'] = (recent_ldr_10[-1] - recent_ldr_10[0]) / len(recent_ldr_10)
                features['ldr_velocity'] = recent_ldr_10[-1] - recent_ldr_10[-2] if len(recent_ldr_10) > 1 else 0
                
                # Acceleration (change in velocity)
                if len(recent_ldr_10) > 2:
                    vel_prev = recent_ldr_10[-2] - recent_ldr_10[-3]
                    vel_current = recent_ldr_10[-1] - recent_ldr_10[-2]
                    features['ldr_acceleration'] = vel_current - vel_prev
                else:
                    features['ldr_acceleration'] = 0
            else:
                features['ldr_trend'] = 0
                features['ldr_velocity'] = 0
                features['ldr_acceleration'] = 0
        else:
            # Default values when no history
            for stat in ['ldr_mean_10', 'ldr_std_10', 'ldr_max_10', 'ldr_min_10',
                        'ldr_mean_30', 'ldr_std_30', 'ldr_max_30', 'ldr_min_30',
                        'ldr_trend', 'ldr_velocity', 'ldr_acceleration']:
                features[stat] = 0
            features['ldr_mean_10'] = ldr_value
            features['ldr_mean_30'] = ldr_value
        
        # Enhanced weather features
        features['temp'] = weather_data.get('temp', 25.0)
        features['humidity'] = weather_data.get('humidity', 60.0)
        features['clouds'] = weather_data.get('clouds', 20.0)
        features['pressure'] = weather_data.get('pressure', 1013.0)
        features['visibility'] = weather_data.get('visibility', 10000.0)
        
        # Weather condition encoding
        weather_desc = weather_data.get('description', 'clear').lower()
        features['weather_clear'] = 1 if any(word in weather_desc for word in ['clear', 'sunny']) else 0
        features['weather_cloudy'] = 1 if any(word in weather_desc for word in ['cloud', 'overcast', 'fog', 'mist']) else 0
        features['weather_rain'] = 1 if any(word in weather_desc for word in ['rain', 'drizzle', 'shower']) else 0
        features['weather_extreme'] = 1 if any(word in weather_desc for word in ['storm', 'thunder', 'extreme']) else 0
        
        # Interaction features
        features['hour_x_clouds'] = features['hour'] * features['clouds'] / 100.0
        features['ldr_x_temp'] = features['ldr_normalized'] * features['temp']
        features['temp_x_humidity'] = features['temp'] * features['humidity'] / 100.0
        
        # Ensure all features are present
        for feature_name in self.feature_names:
            if feature_name not in features:
                features[feature_name] = 0.0
        
        return features
    
    def features_to_array(self, features_dict):
        """Convert features dict to numpy array with consistent ordering"""
        return np.array([features_dict[k] for k in self.feature_names]).reshape(1, -1)

# ============================================================
# ENHANCED ML MODEL MANAGER
# ============================================================
class EnhancedMLModelManager:
    """Enhanced ML model manager with better performance tracking"""
    
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self.pipelines = {}
        self.metadata = {}
        self.feature_engine = EnhancedFeatureEngine()
        self.residuals_summary = {}
        self.performance_history = deque(maxlen=100)
        self.load_models()
        
    def create_pipeline(self, model_name='random_forest'):
        """Create sklearn Pipeline with preprocessing and model"""
        n_jobs = 1  # Single job to prevent CPU contention
        
        if model_name == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=n_jobs
            )
        elif model_name == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        elif model_name == 'ridge':
            model = Ridge(alpha=1.0, random_state=42)
        else:
            model = RandomForestRegressor(random_state=42, n_jobs=n_jobs)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        return pipeline
    
    def train_with_time_series_cv(self, X, y, model_name='random_forest', n_splits=5, tune_hyperparameters=False):
        """Enhanced training with better validation and performance tracking"""
        logger.info(f"Training {model_name} with TimeSeriesSplit (n_splits={n_splits})")
        
        training_start_time = time.time()
        tscv = TimeSeriesSplit(n_splits=n_splits)
        pipeline = self.create_pipeline(model_name)
        
        cv_scores = {'mae': [], 'rmse': [], 'r2': []}
        fold_metrics = []
        all_residuals = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            residuals = y_val - y_pred
            all_residuals.extend(residuals)
            
            cv_scores['mae'].append(mae)
            cv_scores['rmse'].append(rmse)
            cv_scores['r2'].append(r2)
            
            fold_metrics.append({
                'fold': fold,
                'mae': mae,
                'rmse': rmse, 
                'r2': r2,
                'train_size': len(X_train),
                'val_size': len(X_val)
            })
            
            logger.info(f"  Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
        
        # Enhanced residual analysis
        if all_residuals:
            residuals = np.array(all_residuals)
            model_key = f"{model_name}_latest"
            self.residuals_summary[model_key] = {
                'residual_count': len(residuals),
                'residual_mean': float(np.mean(residuals)),
                'residual_std': float(np.std(residuals)),
                'residual_skew': float(pd.Series(residuals).skew()),
                'q95': float(np.quantile(np.abs(residuals), 0.95)),
                'q99': float(np.quantile(np.abs(residuals), 0.99)),
                'residual_range': [float(np.min(residuals)), float(np.max(residuals))],
                'outlier_count': int(np.sum(np.abs(residuals) > 3 * np.std(residuals)))
            }
        
        # Hyperparameter tuning
        if tune_hyperparameters and model_name == 'random_forest':
            logger.info("Running enhanced hyperparameter tuning...")
            param_distributions = {
                'model__n_estimators': [100, 150, 200],
                'model__max_depth': [15, 20, 25, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
            
            random_search = RandomizedSearchCV(
                pipeline,
                param_distributions,
                n_iter=15,
                cv=tscv,
                scoring='neg_mean_absolute_error',
                random_state=42,
                n_jobs=1
            )
            
            random_search.fit(X, y)
            pipeline = random_search.best_estimator_
            logger.info(f"Best params: {random_search.best_params_}")
            logger.info(f"Best score: {-random_search.best_score_:.4f}")
        else:
            # Final training on all data
            pipeline.fit(X, y)
        
        model_key = f"{model_name}_latest"
        
        # Enhanced metadata
        self.pipelines[model_key] = pipeline
        self.metadata[model_key] = {
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'cv_mae_mean': float(np.mean(cv_scores['mae'])),
            'cv_mae_std': float(np.std(cv_scores['mae'])),
            'cv_rmse_mean': float(np.mean(cv_scores['rmse'])),
            'cv_r2_mean': float(np.mean(cv_scores['r2'])),
            'fold_metrics': fold_metrics,
            'feature_names': self.feature_engine.feature_names,
            'feature_importance': self.get_feature_importance(pipeline, model_name),
            'model_key': model_key,
            'residuals_available': len(all_residuals) > 0,
            'training_time_seconds': time.time() - training_start_time
        }
        
        # Track performance history
        self.performance_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_key': model_key,
            'cv_mae': float(np.mean(cv_scores['mae'])),
            'cv_rmse': float(np.mean(cv_scores['rmse']))
        })
        
        logger.info(f"Training complete. CV MAE: {np.mean(cv_scores['mae']):.4f} ± {np.std(cv_scores['mae']):.4f}")
        
        return pipeline, self.metadata[model_key]
    
    def get_feature_importance(self, pipeline, model_name):
        """Get feature importance if available"""
        if model_name == 'ridge':
            try:
                return pipeline.named_steps['model'].coef_.tolist()
            except:
                return []
        elif model_name in ['random_forest', 'gradient_boosting']:
            try:
                return pipeline.named_steps['model'].feature_importances_.tolist()
            except:
                return []
        return []
    
    def save_model(self, model_name='random_forest', version='latest'):
        """Enhanced model saving with versioning"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_key = f"{model_name}_{version}"
        timestamped_key = f"{model_name}_{timestamp}"
        
        if model_key not in self.pipelines:
            logger.error(f"Model {model_key} not found in pipelines")
            return False
        
        for save_key in [model_key, timestamped_key]:
            model_path = os.path.join(self.model_dir, f'{save_key}.joblib')
            
            try:
                with tempfile.NamedTemporaryFile(mode='wb', delete=False, 
                                               dir=self.model_dir, 
                                               suffix='.tmp') as tmp_file:
                    tmp_path = tmp_file.name
                    joblib.dump({
                        'pipeline': self.pipelines[model_key],
                        'metadata': self.metadata[model_key],
                        'feature_names': self.feature_engine.feature_names,
                        'residuals_summary': self.residuals_summary.get(model_key, {})
                    }, tmp_path, compress=3)
                
                os.replace(tmp_path, model_path)
                logger.info(f"Model saved: {model_path}")
                
            except Exception as e:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                logger.error(f"Error saving model {save_key}: {e}")
                return False
        
        return True
    
    def load_models(self):
        """Enhanced model loading"""
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory {self.model_dir} not found")
            return
        
        loaded_count = 0
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.joblib'):
                model_path = os.path.join(self.model_dir, filename)
                model_key = filename.replace('.joblib', '')
                
                try:
                    saved_data = joblib.load(model_path)
                    self.pipelines[model_key] = saved_data['pipeline']
                    self.metadata[model_key] = saved_data['metadata']
                    
                    if 'residuals_summary' in saved_data:
                        self.residuals_summary[model_key] = saved_data['residuals_summary']
                    
                    if 'feature_names' in saved_data:
                        self.feature_engine.feature_names = saved_data['feature_names']
                    
                    loaded_count += 1
                    logger.info(f"Loaded model: {model_key}")
                except Exception as e:
                    logger.error(f"Error loading {model_key}: {e}")
        
        logger.info(f"Successfully loaded {loaded_count} models")
    
    def predict(self, features_dict, model_name='random_forest_latest'):
        """Enhanced prediction with confidence intervals"""
        if model_name not in self.pipelines:
            available_models = list(self.pipelines.keys())
            logger.warning(f"Model {model_name} not found. Available: {available_models}")
            if available_models:
                model_name = available_models[0]
            else:
                return None
        
        try:
            X = self.feature_engine.features_to_array(features_dict)
            pipeline = self.pipelines[model_name]
            prediction = pipeline.predict(X)[0]
            
            # Enhanced confidence intervals
            if model_name in self.residuals_summary:
                residual_data = self.residuals_summary[model_name]
                margin = residual_data['q95']
                confidence = max(0.1, 1 - (margin * 1.5))  # More realistic confidence
            else:
                cv_mae = self.metadata[model_name].get('cv_mae_mean', 0.1)
                margin = 1.96 * cv_mae
                confidence = max(0.1, 1 - (cv_mae * 2))
            
            lower_bound = max(0, prediction - margin)
            upper_bound = min(1, prediction + margin)
            
            return {
                'prediction': float(prediction),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence': float(confidence),
                'margin': float(margin),
                'model_used': model_name
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

# Initialize enhanced model manager
model_manager = EnhancedMLModelManager()

# ============================================================
# ENHANCED MONITORING & DRIFT DETECTION
# ============================================================
class EnhancedModelMonitor:
    """Enhanced monitoring with statistical testing"""
    
    def __init__(self):
        self.predictions = deque(maxlen=2000)
        self.actuals = deque(maxlen=2000)
        self.feature_distributions = {}
        self.baseline_features = None
        self.performance_alerts = deque(maxlen=50)
        
    def log_prediction(self, features, prediction, actual=None):
        """Enhanced prediction logging"""
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
            
            # Check for performance degradation
            self.check_performance_degradation()
        
        # Update feature distributions with statistical tracking
        for key, value in features.items():
            if key not in self.feature_distributions:
                self.feature_distributions[key] = {
                    'values': deque(maxlen=1000),
                    'mean': 0,
                    'std': 0,
                    'min': float('inf'),
                    'max': float('-inf')
                }
            
            dist = self.feature_distributions[key]
            dist['values'].append(value)
            values_list = list(dist['values'])
            
            if len(values_list) > 10:
                dist['mean'] = np.mean(values_list)
                dist['std'] = np.std(values_list)
                dist['min'] = min(values_list)
                dist['max'] = max(values_list)
    
    def check_performance_degradation(self):
        """Check for model performance degradation"""
        if len(self.actuals) < 20:
            return
        
        recent_size = min(100, len(self.actuals))
        recent_preds = [p['prediction'] for p in list(self.predictions)[-recent_size:]]
        recent_actuals = [a['value'] for a in list(self.actuals)[-recent_size:]]
        
        current_mae = mean_absolute_error(recent_actuals, recent_preds)
        
        # Compare with historical performance if available
        if len(self.actuals) >= 200:
            historical_size = min(200, len(self.actuals) - recent_size)
            historical_preds = [p['prediction'] for p in list(self.predictions)[-historical_size-recent_size:-recent_size]]
            historical_actuals = [a['value'] for a in list(self.actuals)[-historical_size-recent_size:-recent_size]]
            
            historical_mae = mean_absolute_error(historical_actuals, historical_preds)
            
            # Alert if performance degrades by more than 50%
            if current_mae > historical_mae * 1.5:
                alert = {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'type': 'performance_degradation',
                    'current_mae': current_mae,
                    'historical_mae': historical_mae,
                    'degradation_ratio': current_mae / historical_mae
                }
                self.performance_alerts.append(alert)
                logger.warning(f"Performance degradation detected: {alert}")
    
    def calculate_drift(self):
        """Enhanced drift detection with statistical testing"""
        if not self.baseline_features:
            self.baseline_features = {}
            for feature, dist in self.feature_distributions.items():
                if len(dist['values']) > 50:
                    self.baseline_features[feature] = {
                        'mean': dist['mean'],
                        'std': dist['std'],
                        'n_samples': len(dist['values'])
                    }
            return {}
        
        drift_metrics = {}
        for feature, current_dist in self.feature_distributions.items():
            if feature not in self.baseline_features or len(current_dist['values']) < 50:
                continue
            
            baseline = self.baseline_features[feature]
            current_mean = current_dist['mean']
            baseline_mean = baseline['mean']
            pooled_std = np.sqrt((current_dist['std']**2 + baseline['std']**2) / 2)
            
            if pooled_std > 0:
                # Calculate standardized drift score
                drift_score = abs(current_mean - baseline_mean) / pooled_std
                drift_metrics[feature] = {
                    'drift_score': float(drift_score),
                    'current_mean': float(current_mean),
                    'baseline_mean': float(baseline_mean),
                    'change_pct': float((current_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0,
                    'severity': 'high' if drift_score > 2 else 'medium' if drift_score > 1 else 'low'
                }
        
        return drift_metrics
    
    def get_performance_metrics(self):
        """Enhanced performance metrics"""
        if len(self.actuals) < 5:
            return {}
        
        recent_preds = [p['prediction'] for p in list(self.predictions)[-len(self.actuals):]]
        recent_actuals = [a['value'] for a in self.actuals]
        
        mae = mean_absolute_error(recent_actuals, recent_preds)
        rmse = np.sqrt(mean_squared_error(recent_actuals, recent_preds))
        r2 = r2_score(recent_actuals, recent_preds)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'n_samples': len(recent_actuals),
            'prediction_bias': float(np.mean(np.array(recent_preds) - np.array(recent_actuals)))
        }

monitor = EnhancedModelMonitor()

# ============================================================
# ENHANCED RATE LIMITING
# ============================================================
class EnhancedRateLimiter:
    def __init__(self, max_requests=100, window=3600):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}
        self.lock = Lock()
        self.cleanup_interval = 300  # Clean every 5 minutes
        self.last_cleanup = time.time()
    
    def cleanup_old_entries(self):
        """Clean up old rate limiting entries"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        with self.lock:
            for identifier in list(self.requests.keys()):
                queue = self.requests[identifier]
                while queue and queue[0] < now - self.window:
                    queue.popleft()
                if not queue:
                    del self.requests[identifier]
        
        self.last_cleanup = now
    
    def is_allowed(self, identifier, cost=1):
        """Enhanced rate limiting with request cost"""
        self.cleanup_old_entries()
        
        with self.lock:
            now = time.time()
            if identifier not in self.requests:
                self.requests[identifier] = deque(maxlen=self.max_requests * 2)
            
            queue = self.requests[identifier]
            
            # Clean old requests for this identifier
            while queue and queue[0] < now - self.window:
                queue.popleft()
            
            # Check if under limit
            if len(queue) + cost <= self.max_requests:
                for _ in range(cost):
                    queue.append(now)
                return True, self.max_requests - len(queue)
            return False, 0

enhanced_rate_limiter = EnhancedRateLimiter(max_requests=100, window=3600)

def check_rate_limit(identifier, limit=100, window=3600, cost=1):
    """Enhanced rate limiting with cost"""
    return enhanced_rate_limiter.is_allowed(identifier, cost)

# ============================================================
# ENHANCED WEATHER SERVICE
# ============================================================
def fetch_enhanced_weather(lat=None, lon=None, store_history=True):
    """Enhanced weather fetching with better error handling"""
    use_lat = lat if lat else LOCATION_LAT
    use_lon = lon if lon else LOCATION_LON
    
    cache_key = f"{use_lat},{use_lon}"
    
    cached_data = weather_cache.get(cache_key)
    if cached_data is not None:
        return cached_data
    
    if not OWM_KEY:
        logger.warning("OWM_KEY not set, using enhanced fallback weather data")
        # Enhanced fallback with seasonal variation
        current_month = datetime.now(timezone.utc).month
        if current_month in [12, 1, 2]:  # Winter
            default_weather = {"description": "clear sky", "temp": 22, "humidity": 45, "clouds": 10, "pressure": 1015, "visibility": 10000}
        elif current_month in [3, 4, 5]:  # Summer
            default_weather = {"description": "clear sky", "temp": 32, "humidity": 40, "clouds": 20, "pressure": 1010, "visibility": 8000}
        elif current_month in [6, 7, 8, 9]:  # Monsoon
            default_weather = {"description": "scattered clouds", "temp": 28, "humidity": 75, "clouds": 60, "pressure": 1008, "visibility": 5000}
        else:  # Post-monsoon
            default_weather = {"description": "few clouds", "temp": 26, "humidity": 65, "clouds": 30, "pressure": 1012, "visibility": 7000}
        
        weather_cache.set(cache_key, default_weather)
        return default_weather
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={use_lat}&lon={use_lon}&appid={OWM_KEY}&units=metric"
        resp = requests.get(url, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            result = {
                "description": data["weather"][0]["description"],
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "clouds": data["clouds"]["all"],
                "pressure": data["main"]["pressure"],
                "visibility": data.get("visibility", 10000),
                "wind_speed": data["wind"]["speed"] if "wind" in data else 0
            }
            weather_cache.set(cache_key, result)
            return result
        else:
            logger.error(f"Weather API returned status {resp.status_code}: {resp.text}")
    except requests.exceptions.Timeout:
        logger.error("Weather API request timed out")
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
    
    # Enhanced fallback on error
    fallback_weather = {"description": "clear sky", "temp": 25, "humidity": 60, "clouds": 20, "pressure": 1013, "visibility": 10000}
    weather_cache.set(cache_key, fallback_weather)
    return fallback_weather

# ============================================================
# ENHANCED SIMULATION FUNCTION (FIXED)
# ============================================================
def generate_realistic_solar_data(hours=24, base_ts=None):
    """Generate realistic solar data with proper patterns"""
    if base_ts is None:
        base_ts = int(time.time())
    
    simulated_data = []
    
    for hour_offset in range(hours):
        ts = base_ts + (hour_offset * 3600)
        dt = datetime.fromtimestamp(ts, timezone.utc)
        hour = dt.hour
        minute = dt.minute
        
        # Enhanced realistic solar pattern
        if 6 <= hour <= 18:  # Daylight hours
            # Solar intensity follows a realistic curve peaking at noon
            peak_hour = 12
            distance_from_peak = abs(hour - peak_hour)
            
            # Base intensity curve (more realistic)
            if hour <= 12:
                # Morning: gradual increase
                progress = (hour - 6) / 6.0  # 6 AM to 12 PM
                base_intensity = 800 * (1 - math.exp(-3 * progress))  # Exponential growth
            else:
                # Afternoon: gradual decrease
                progress = (18 - hour) / 6.0  # 12 PM to 6 PM
                base_intensity = 800 * (1 - math.exp(-3 * progress))  # Exponential decay
            
            # Add time-based variation (more intense in middle of hour)
            minute_factor = 1.0 + 0.1 * math.sin(2 * math.pi * minute / 60)
            
            # Add realistic noise (less variation during stable hours)
            if 9 <= hour <= 15:
                noise = (hash(f"{ts}") % 50) - 25  # -25 to +25 during peak
            else:
                noise = (hash(f"{ts}") % 80) - 40  # -40 to +40 during transition
            
            ldr = int(max(100, min(1023, base_intensity * minute_factor + noise)))
            
        else:  # Night hours
            # Very low values at night with minimal sensor noise
            ldr = 15 + (hash(f"{ts}") % 20)  # 15-35 at night
        
        simulated_data.append({
            "timestamp": ts,
            "time_iso": dt.isoformat(),
            "ldr": ldr,
            "deviceId": "simulated-device",
            "hour": hour,
            "minute": minute,
            "date": dt.strftime("%Y-%m-%d")
        })
    
    return simulated_data

# ============================================================
# FLASK APP SETUP
# ============================================================
app = Flask(__name__, template_folder='templates', static_folder='static')

# Enhanced CORS configuration
CORS(app, origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5000", 
    "http://127.0.0.1:5000",
    "http://0.0.0.0:5000",
    "http://192.168.29.229:5000"
], methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Content-Type", "X-API-Key"])

# ============================================================
# ROOT ENDPOINT
# ============================================================
@app.route('/', methods=['GET'])
def home():
    """Serve the dashboard HTML"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Failed to render index.html: {e}")
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Solar Energy Monitoring</title></head>
        <body>
            <h1>Solar Energy Monitoring System</h1>
            <p>API is running. Use endpoints:</p>
            <ul>
                <li>POST /api/data/upload - Upload sensor data</li>
                <li>GET /api/data/readings - Get readings</li>
                <li>GET /api/health - Health check</li>
            </ul>
        </body>
        </html>
        """

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "solar-energy-scheduler"
    })

@app.route('/api/health', methods=['GET'])
def api_health_check():
    """API health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "solar-energy-monitoring",
        "version": "2.0"
    })

# ============================================================
# AUTHENTICATION
# ============================================================
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = (
            request.headers.get("X-API-Key")
            or request.headers.get("x-api-key")
            or request.args.get("api_key")
            or (request.get_json(silent=True) or {}).get("api_key")
        )

        logger.info(f"DEBUG: API key received = {provided_key}")

        if not provided_key:
            return jsonify({
                "error": "Unauthorized",
                "message": "API key missing"
            }), 401

        if provided_key != API_KEY:
            return jsonify({
                "error": "Unauthorized",
                "message": "Valid API key required"
            }), 401

        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# DATABASE MANAGEMENT
# ============================================================
def get_db():
    """Get database connection"""
    if 'db' not in g:
        g.db = sqlite3.connect(DB, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA journal_mode=WAL')
        g.db.execute('PRAGMA synchronous=NORMAL')
        g.db.execute('PRAGMA cache_size=-64000')
    return g.db

def get_db_threadsafe():
    """Get thread-safe database connection"""
    db = sqlite3.connect(DB, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    return db

def close_db(error):
    """Close database connection"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db_threadsafe()
    c = db.cursor()

    # Create readings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            ldr INTEGER NOT NULL CHECK (ldr >= 0 AND ldr <= 1023),
            temperature REAL,
            humidity REAL,
            rssi INTEGER,
            deviceId TEXT NOT NULL,
            created_at DATETIME DEFAULT (datetime('now'))
        )
    ''')

    # Create indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON readings(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_deviceId ON readings(deviceId)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp_deviceId ON readings(timestamp, deviceId)')

    db.commit()
    db.close()

# ============================================================
# ✅ FIXED: VALIDATION FUNCTION MOVED HERE (BEFORE UPLOAD ENDPOINT)
# ============================================================
def validate_sensor_data(data):
    """Enhanced validation for sensor data"""
    if not isinstance(data, dict):
        return False, "Invalid data format"
    
    # Check required fields
    if 'ldr' not in data:
        return False, "LDR value is required"
    
    # Validate LDR range
    ldr = data['ldr']
    if not isinstance(ldr, (int, float)):
        return False, "LDR must be a number"
    if ldr < 0 or ldr > 1023:
        return False, "LDR must be between 0 and 1023"
    
    # Validate deviceId if present
    if 'deviceId' in data and not isinstance(data['deviceId'], str):
        return False, "deviceId must be a string"
    
    # Validate optional fields
    if 'temperature' in data and not isinstance(data['temperature'], (int, float, type(None))):
        return False, "Temperature must be a number or null"
    
    if 'humidity' in data and not isinstance(data['humidity'], (int, float, type(None))):
        return False, "Humidity must be a number or null"
    
    if 'rssi' in data and not isinstance(data['rssi'], (int, float, type(None))):
        return False, "RSSI must be a number or null"
    
    return True, "Valid"

# ============================================================
# API ENDPOINTS
# ============================================================
@app.route('/api/data/upload', methods=['POST'])
def upload_sensor_data():
    """Upload sensor data with enhanced validation"""
    client_id = request.remote_addr
    allowed, remaining = check_rate_limit(client_id, limit=60, window=3600, cost=1)
    if not allowed:
        return jsonify({"error": "Rate limit exceeded"}), 429

    try:
        logger.info("UPLOAD ENDPOINT HIT SUCCESSFULLY")

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # ✅ NOW THIS WILL WORK - validate_sensor_data is defined above
        is_valid, message = validate_sensor_data(data)
        if not is_valid:
            return jsonify({"error": message}), 400

        db = get_db()
        c = db.cursor()

        timestamp = data.get('timestamp', int(time.time()))
        ldr = data['ldr']
        device_id = data.get('deviceId', 'default-device')
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        rssi = data.get('rssi')

        c.execute(
            """
            INSERT INTO readings (timestamp, ldr, temperature, humidity, rssi, deviceId)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, ldr, temperature, humidity, rssi, device_id)
        )

        db.commit()

        response_data = {
            "status": "success",
            "message": "Sensor data stored successfully",
            "data": {
                "id": c.lastrowid,
                "timestamp": timestamp,
                "ldr": ldr,
                "temperature": temperature,
                "humidity": humidity,
                "rssi": rssi,
                "deviceId": device_id
            }
        }

        logger.info(
            f"Stored sensor data from {device_id}: "
            f"LDR={ldr}, TEMP={temperature}, HUM={humidity}"
        )

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error uploading sensor data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/data/readings', methods=['GET'])
def get_readings():
    """Get sensor readings"""
    try:
        db = get_db()
        c = db.cursor()
        
        limit = request.args.get('limit', 100, type=int)
        device_id = request.args.get('deviceId')
        hours_back = request.args.get('hours', type=int)
        
        query = "SELECT * FROM readings WHERE 1=1"
        params = []
        
        if device_id:
            query += " AND deviceId = ?"
            params.append(device_id)
        
        if hours_back:
            cutoff_ts = int(time.time()) - (hours_back * 3600)
            query += " AND timestamp >= ?"
            params.append(cutoff_ts)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        c.execute(query, params)
        readings = [dict(row) for row in c.fetchall()]
        
        return jsonify({
            "status": "success",
            "count": len(readings),
            "readings": readings
        })
        
    except Exception as e:
        logger.error(f"Error fetching readings: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/weather', methods=['GET'])
def get_weather():
    try:
        weather = fetch_enhanced_weather()
        return jsonify(weather)
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return jsonify({"error": "Failed to fetch weather"}), 500

# ============================================================
# APPLICATION STARTUP
# ============================================================
def initialize_system():
    """Initialize system"""
    logger.info("Initializing enhanced solar energy scheduler...")
    init_db()

if __name__ == '__main__':
    initialize_system()
    
    print("\n" + "="*70)
    print("🚀 ENHANCED SMART SOLAR ENERGY SCHEDULER - PRODUCTION READY")
    print("="*70)
    print(f"\n🎯 Server starting on http://0.0.0.0:{PORT}")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)
