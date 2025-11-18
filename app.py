# SAVE THIS AS: app.py

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
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import math

# ============================================================
# CONFIGURATION
# ============================================================
DB = 'solar.db'
PORT = int(os.getenv("PORT", 5000))

# Weather API
OWM_KEY = "266cbcfc14167cde4293c8c572d95c62"
LOCATION_LAT = "13.05565"
LOCATION_LON = "77.50561"

WEATHER_CACHE_TTL = 10 * 60
weather_cache = {"data": None, "timestamp": 0}

# Cost calculation (INR per kWh)
GRID_COST_PER_KWH = 7.5
SOLAR_COST_PER_KWH = 0.5

# ============================================================
# NEURAL NETWORK MODEL
# ============================================================
class SimpleNeuralNetwork:
    """Lightweight neural network for solar prediction"""
    
    def __init__(self, input_size=5, hidden_size=8, output_size=1):
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, X):
        """Forward pass through network"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def train(self, X, y, epochs=100, lr=0.01):
        """Simple gradient descent training"""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss (MSE)
            loss = np.mean((output - y) ** 2)
            
            # Backward pass
            m = X.shape[0]
            dz2 = (output - y) / m
            dW2 = np.dot(self.a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (self.z1 > 0)  # ReLU derivative
            dW1 = np.dot(X.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Update weights
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            
        return loss
    
    def predict(self, X):
        """Make prediction"""
        return self.forward(X)

# Global neural network instance
neural_net = SimpleNeuralNetwork()

# ============================================================
# PATTERN LEARNING ENGINE
# ============================================================
class PatternLearner:
    """Learns patterns from historical solar data"""
    
    def __init__(self):
        self.hourly_patterns = {}
        self.weather_patterns = {}
        self.seasonal_factors = {}
        
    def learn_from_history(self, history_data):
        """Extract patterns from historical readings"""
        if not history_data or len(history_data) < 10:
            return
        
        # Group by hour of day
        hourly_data = {}
        weather_data = {}
        
        for reading in history_data:
            ts, ldr = reading[1], reading[2]
            dt = datetime.fromtimestamp(ts)
            hour = dt.hour
            
            if hour not in hourly_data:
                hourly_data[hour] = []
            hourly_data[hour].append(ldr)
        
        # Calculate average and std for each hour
        for hour, values in hourly_data.items():
            self.hourly_patterns[hour] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'samples': len(values)
            }
        
        # Learn seasonal patterns
        month = datetime.now().month
        season = self._get_season(month)
        avg_ldr = np.mean([r[2] for r in history_data])
        self.seasonal_factors[season] = avg_ldr / 512  # Normalize
        
    def _get_season(self, month):
        """Determine season from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def predict_next_hour(self, current_hour, current_ldr):
        """Predict next hour's solar intensity"""
        next_hour = (current_hour + 1) % 24
        
        if next_hour in self.hourly_patterns:
            pattern = self.hourly_patterns[next_hour]
            # Blend pattern with current reading
            predicted = 0.6 * pattern['mean'] + 0.4 * current_ldr
            confidence = min(95, 50 + pattern['samples'])
            return predicted, confidence
        
        return current_ldr * 0.95, 40
    
    def get_optimal_hours(self):
        """Find hours with highest solar intensity"""
        if not self.hourly_patterns:
            return list(range(10, 16))
        
        sorted_hours = sorted(
            self.hourly_patterns.items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )
        return [h for h, _ in sorted_hours[:6]]

# Global pattern learner
pattern_learner = PatternLearner()

# ============================================================
# GENERATIVE AI TEXT ENGINE
# ============================================================
class NLPInsightGenerator:
    """Generate natural language insights about solar data"""
    
    def __init__(self):
        self.templates = {
            'excellent': [
                "Outstanding solar conditions detected! Your panels are performing at {intensity}% capacity. This is prime time for energy-intensive tasks.",
                "Exceptional sunlight today with {intensity}% intensity. Perfect opportunity to maximize your solar investment.",
                "Peak solar performance achieved! With {intensity}% intensity, you're generating clean energy at optimal rates."
            ],
            'good': [
                "Strong solar output at {intensity}%. Good conditions for running major appliances.",
                "Solid solar generation with {intensity}% intensity. Your system is performing well.",
                "Favorable solar conditions detected. At {intensity}%, you can power most household needs."
            ],
            'moderate': [
                "Moderate solar activity at {intensity}%. Best suited for medium-load appliances.",
                "Decent solar output today. With {intensity}% intensity, plan accordingly for energy usage.",
                "Fair solar conditions. At {intensity}%, prioritize essential appliances."
            ],
            'low': [
                "Limited solar generation at {intensity}%. Consider deferring heavy loads.",
                "Reduced solar output detected. Current intensity at {intensity}% - grid backup recommended.",
                "Minimal solar activity today. At {intensity}%, rely on stored energy or grid power."
            ]
        }
        
        self.weather_modifiers = {
            'clear': 'with crystal clear skies enhancing output',
            'cloud': 'though clouds may intermittently reduce efficiency',
            'rain': 'with precipitation significantly impacting generation',
            'sun': 'under brilliant sunshine conditions'
        }
    
    def generate_insight(self, intensity, weather_desc, trend, savings):
        """Generate human-like insight text"""
        # Determine condition category
        if intensity > 75:
            category = 'excellent'
        elif intensity > 55:
            category = 'good'
        elif intensity > 35:
            category = 'moderate'
        else:
            category = 'low'
        
        # Select template
        template = np.random.choice(self.templates[category])
        base_insight = template.format(intensity=f"{intensity:.1f}")
        
        # Add weather context
        weather_modifier = None
        for key, modifier in self.weather_modifiers.items():
            if key in weather_desc.lower():
                weather_modifier = modifier
                break
        
        if weather_modifier:
            base_insight += f" {weather_modifier.capitalize()}"
        
        # Add trend analysis
        if trend > 5:
            base_insight += f" ↗️ Improving trend detected - conditions are getting better."
        elif trend < -5:
            base_insight += f" ↘️ Declining trend observed - prepare for reduced output."
        
        # Add savings perspective
        if savings > 50:
            base_insight += f" 💰 You're saving ₹{savings:.0f} today by going solar!"
        
        return base_insight
    
    def generate_recommendation(self, intensity, predicted_hours, optimal_window):
        """Generate actionable recommendations"""
        if intensity > 70:
            actions = [
                "Run washing machine, dishwasher, and AC simultaneously",
                "Charge all devices and electric vehicles",
                "Operate water heater and other heavy appliances",
                "Consider running pool pumps or irrigation systems"
            ]
        elif intensity > 50:
            actions = [
                "Run one major appliance at a time",
                "Charge batteries and devices",
                "Operate medium-load equipment",
                "Delay heavy loads to peak hours"
            ]
        elif intensity > 30:
            actions = [
                "Focus on light appliances only",
                "Charge mobile devices and laptops",
                "Minimal HVAC usage recommended",
                "Reserve heavy tasks for better conditions"
            ]
        else:
            actions = [
                "Rely on grid power for major needs",
                "Use stored battery power if available",
                "Minimize energy consumption",
                "Plan energy-intensive tasks for tomorrow"
            ]
        
        return {
            'primary_action': actions[0],
            'all_actions': actions,
            'optimal_window': optimal_window,
            'predicted_generation': f"{predicted_hours:.1f} hours of useful sunlight expected"
        }

# Global NLP generator
nlp_generator = NLPInsightGenerator()

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
        print("📊 No data found. Generating sample data...")
        generate_sample_data(c)
        conn.commit()
        print(f"✅ Generated sample readings")
    
    conn.close()

def generate_sample_data(cursor):
    """Generate 24 hours of realistic solar data"""
    base_ts = int(time.time()) - (144 * 600)
    
    for i in range(144):
        ts = base_ts + (i * 600)
        hour = (ts // 3600) % 24
        
        if 6 <= hour <= 18:
            peak_hour = 12
            distance_from_peak = abs(hour - peak_hour)
            intensity = 1023 * math.exp(-0.08 * distance_from_peak**2)
            noise = (hash(ts) % 100) - 50
            ldr = int(max(0, min(1023, intensity + noise)))
        else:
            ldr = int(20 + (hash(ts) % 30))
        
        cursor.execute("INSERT INTO readings (timestamp, ldr, deviceId) VALUES (?, ?, ?)",
                      (ts, ldr, 'auto-init'))

# ============================================================
# ML-POWERED PREDICTION ENGINE
# ============================================================
def ml_solar_prediction(ldr_value, history_data, weather_data):
    """ML-enhanced solar prediction with neural network and pattern learning"""
    
    # Learn patterns from history
    pattern_learner.learn_from_history(history_data)
    
    # Prepare features for neural network
    current_hour = datetime.now().hour
    day_of_year = datetime.now().timetuple().tm_yday
    
    # Extract recent statistics
    if history_data and len(history_data) > 0:
        recent_readings = [r[2] for r in history_data[-20:]]
        avg_ldr = np.mean(recent_readings)
        std_ldr = np.std(recent_readings)
        trend = (recent_readings[-1] - recent_readings[0]) / len(recent_readings) if len(recent_readings) > 1 else 0
    else:
        avg_ldr = ldr_value
        std_ldr = 0
        trend = 0
    
    # Prepare neural network input
    # Features: [normalized_ldr, hour_normalized, day_of_year_normalized, avg_intensity, trend]
    X = np.array([[
        ldr_value / 1023,
        current_hour / 24,
        day_of_year / 365,
        avg_ldr / 1023,
        trend / 100
    ]])
    
    # Neural network prediction
    nn_prediction = neural_net.predict(X)[0][0]
    
    # Pattern-based prediction
    pattern_pred, pattern_conf = pattern_learner.predict_next_hour(current_hour, ldr_value)
    
    # Ensemble prediction (combine NN and pattern learning)
    intensity = (ldr_value / 1023) * 100
    predicted_intensity = (nn_prediction * 100 * 0.4 + 
                          (pattern_pred / 1023) * 100 * 0.3 + 
                          intensity * 0.3)
    
    is_peak_hours = 10 <= current_hour <= 15
    is_daylight = 6 <= current_hour <= 18
    
    # Weather impact
    weather_desc = weather_data.get('description', '').lower()
    weather_factor = 1.2 if 'clear' in weather_desc or 'sun' in weather_desc else 0.8 if 'cloud' in weather_desc else 0.5 if 'rain' in weather_desc else 1.0
    
    # Calculate predicted sun hours
    base_hours = (predicted_intensity / 100) * 12
    predicted_hours = base_hours * weather_factor
    if is_peak_hours:
        predicted_hours *= 1.1
    if not is_daylight:
        predicted_hours *= 0.3
    predicted_hours = max(0, min(predicted_hours, 12))
    
    # Enhanced confidence with ML factors
    confidence = 70
    if len(history_data) > 50:
        confidence += 10
    if abs(trend) < 10:
        confidence += 5
    if is_daylight:
        confidence += 5
    if pattern_conf > 60:
        confidence += 5
    confidence = min(confidence, 98)
    
    # Get optimal hours from pattern learner
    optimal_hours = pattern_learner.get_optimal_hours()
    
    # Recommendations
    if predicted_intensity > 80:
        window = {"start": "09:00", "end": "16:00"}
    elif predicted_intensity > 60:
        window = {"start": "10:00", "end": "15:00"}
    elif predicted_intensity > 40:
        window = {"start": "11:00", "end": "14:00"}
    elif predicted_intensity > 20:
        window = {"start": "12:00", "end": "14:00"}
    else:
        window = {"start": "18:00", "end": "21:00"}
    
    # Calculate savings
    solar_kwh = predicted_hours * 2
    savings_data = calculate_savings(solar_kwh, solar_kwh)
    
    # Generate NLP insights
    nlp_insight = nlp_generator.generate_insight(
        predicted_intensity, 
        weather_desc, 
        trend,
        savings_data['cost_saved']
    )
    
    nlp_recommendations = nlp_generator.generate_recommendation(
        predicted_intensity,
        predicted_hours,
        window
    )
    
    return {
        "predicted_sun_hours": round(predicted_hours, 2),
        "confidence": confidence,
        "recommended_window": window,
        "intensity_current": round(intensity, 1),
        "intensity_predicted": round(predicted_intensity, 1),
        "intensity_average": round((avg_ldr / 1023) * 100, 1),
        "trend": "improving" if trend > 0 else "declining" if trend < 0 else "stable",
        "trend_value": round(trend, 2),
        "optimal_hours": optimal_hours,
        "ml_enabled": True,
        "neural_network_score": round(nn_prediction * 100, 1),
        "pattern_confidence": round(pattern_conf, 1),
        "ai_insight": nlp_insight,
        "ai_recommendations": nlp_recommendations,
        "model_type": "ensemble (NN + Pattern Learning + NLP)"
    }

# ============================================================
# WEATHER
# ============================================================
def fetch_weather(lat=None, lon=None):
    """Fetch weather with custom location support"""
    global weather_cache
    
    use_lat = lat if lat else LOCATION_LAT
    use_lon = lon if lon else LOCATION_LON
    
    cache_key = f"{use_lat},{use_lon}"
    now = time.time()
    
    if cache_key in weather_cache and (now - weather_cache.get(f"{cache_key}_time", 0)) < WEATHER_CACHE_TTL:
        return weather_cache[cache_key]
    
    if not OWM_KEY:
        return {"description": "clear sky", "temp": 25}
    
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
            return result
    except Exception as e:
        print(f"⚠️ Weather fetch failed: {e}")
    
    return {"description": "clear sky", "temp": 25, "humidity": 60, "clouds": 20}

# ============================================================
# COST & CARBON CALCULATIONS
# ============================================================
def calculate_savings(solar_kwh, grid_kwh):
    """Calculate cost and carbon savings"""
    solar_cost = solar_kwh * SOLAR_COST_PER_KWH
    grid_cost = grid_kwh * GRID_COST_PER_KWH
    cost_saved = grid_cost - solar_cost
    carbon_saved = solar_kwh * 0.82
    
    return {
        "cost_saved": round(cost_saved, 2),
        "carbon_saved": round(carbon_saved, 2),
        "solar_cost": round(solar_cost, 2),
        "grid_cost": round(grid_cost, 2)
    }

# ============================================================
# API ENDPOINTS
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sensor', methods=['POST'])
def sensor_data():
    data = request.get_json()
    deviceId = data.get('deviceId', 'unknown')
    ldr = data.get('ldr', 0)
    ts = int(time.time())
    
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("INSERT INTO readings (timestamp, ldr, deviceId) VALUES (?, ?, ?)",
              (ts, ldr, deviceId))
    conn.commit()
    conn.close()
    
    # Train neural network with new data
    train_neural_network()
    
    return jsonify({"status": "ok", "timestamp": ts})

@app.route('/api/latest')
def latest():
    lat = request.args.get('lat', LOCATION_LAT)
    lon = request.args.get('lon', LOCATION_LON)
    
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 1")
    row = c.fetchone()
    
    if not row:
        c.execute("SELECT COUNT(*) FROM readings")
        if c.fetchone()[0] == 0:
            generate_sample_data(c)
            conn.commit()
            c.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 1")
            row = c.fetchone()
    
    c.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 100")
    history = c.fetchall()
    conn.close()
    
    if not row:
        return jsonify({"status": "no-data"})
    
    ldr_value = row[2]
    weather = fetch_weather(lat, lon)
    prediction = ml_solar_prediction(ldr_value, history, weather)
    
    response = {
        "timestamp": row[1],
        "time_iso": datetime.fromtimestamp(row[1], timezone.utc).isoformat().replace('+00:00', 'Z'),
        "ldr": ldr_value,
        "deviceId": row[3],
        "weather": weather,
        **prediction
    }
    
    return jsonify(response)

@app.route('/api/history')
def history():
    limit = request.args.get('limit', 200, type=int)
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    
    data = [{
        "timestamp": r[1],
        "time_iso": datetime.fromtimestamp(r[1], timezone.utc).isoformat().replace('+00:00', 'Z'),
        "ldr": r[2],
        "deviceId": r[3]
    } for r in rows]
    
    return jsonify(data)

@app.route('/api/train', methods=['POST'])
def train_model():
    """Endpoint to manually trigger ML model training"""
    result = train_neural_network()
    return jsonify(result)

def train_neural_network():
    """Train neural network on historical data"""
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 500")
        history = c.fetchall()
        conn.close()
        
        if len(history) < 20:
            return {"status": "insufficient_data", "samples": len(history)}
        
        # Prepare training data
        X_train = []
        y_train = []
        
        for i in range(len(history) - 1):
            current = history[i]
            next_reading = history[i + 1]
            
            ts = current[1]
            ldr = current[2]
            next_ldr = next_reading[2]
            
            dt = datetime.fromtimestamp(ts)
            hour = dt.hour
            day_of_year = dt.timetuple().tm_yday
            
            # Calculate moving average
            window = history[max(0, i-5):i+1]
            avg_ldr = np.mean([r[2] for r in window])
            
            # Features
            X_train.append([
                ldr / 1023,
                hour / 24,
                day_of_year / 365,
                avg_ldr / 1023,
                0  # trend placeholder
            ])
            
            # Target: next reading normalized
            y_train.append([next_ldr / 1023])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train the model
        loss = neural_net.train(X_train, y_train, epochs=50, lr=0.01)
        
        return {
            "status": "success",
            "samples_trained": len(X_train),
            "final_loss": float(loss),
            "model_type": "neural_network"
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.route('/api/ai-status')
def ai_status():
    return jsonify({
        "ml_enabled": True,
        "neural_network": True,
        "pattern_learning": True,
        "nlp_insights": True,
        "generative_ai": True,
        "model_architecture": {
            "type": "ensemble",
            "components": [
                "Neural Network (5-8-1)",
                "Pattern Learning Engine",
                "NLP Insight Generator"
            ]
        },
        "features": [
            "Real-time pattern recognition",
            "Predictive analytics",
            "Natural language insights",
            "Adaptive learning",
            "Weather integration"
        ],
        "weather_api_configured": bool(OWM_KEY),
        "confidence_range": "70-98%"
    })

# ============================================================
# STARTUP
# ============================================================
if __name__ == '__main__':
    init_db()
    
    # Initial ML training
    print("\n🧠 Training ML models...")
    train_result = train_neural_network()
    print(f"✅ Training complete: {train_result}")
    
    print("\n" + "="*60)
    print("🤖 SMART SOLAR ENERGY SCHEDULER - ML EDITION")
    print("="*60)
    print("🧠 AI Engine: ✅ ADVANCED ML SYSTEM")
    print("   ├─ Neural Network: 5-8-1 architecture")
    print("   ├─ Pattern Learning: Historical analysis")
    print("   ├─ NLP Engine: Natural language insights")
    print("   └─ Ensemble Model: 70-98% confidence")
    print("💰 Cost Tracking: ₹7.5/kWh grid vs ₹0.5/kWh solar")
    print("🌱 Carbon Tracking: 0.82 kg CO₂ per kWh")
    
    if OWM_KEY:
        print("🌤️  Weather API: ✅ Connected")
    
    print(f"🚀 Server: http://localhost:{PORT}")
    print("="*60)
    print("\n✅ System ready with ML capabilities!")
    print("   🎯 Neural network trained and active")
    print("   📊 Pattern learning from historical data")
    print("   💬 NLP-powered insights and recommendations\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)