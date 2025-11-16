# app.py — Claude AI-Powered Smart Solar Scheduler (Exact Predictions)
import os
import time
import sqlite3
import threading
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import requests

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DB = 'solar.db'
PORT = int(os.getenv("PORT", 5000))
OWM_KEY = os.getenv("OPENWEATHER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LOCATION_LAT = os.getenv("LAT", "13.05565")
LOCATION_LON = os.getenv("LON", "77.50561")
WEATHER_CACHE_TTL = 10 * 60

# ------------------------------------------------------------
# FLASK APP INITIALIZATION
# ------------------------------------------------------------
app = Flask(__name__, instance_path=os.path.join(os.getcwd(), 'instance'))
CORS(app)

@app.route('/favicon.ico')
def favicon_noop():
    return ('', 204)

# ------------------------------------------------------------
# DATABASE SETUP
# ------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deviceId TEXT,
            ts INTEGER,
            ldr INTEGER
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS ai_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER,
            prediction_data TEXT,
            actual_data TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------
# CLAUDE AI PREDICTION ENGINE
# ------------------------------------------------------------
class ClaudeAIPredictionEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.enabled = bool(api_key)
        self.cache = {"ts": 0, "data": None, "key": None}
        self.cache_ttl = 300  # 5 minutes
    
    def predict_solar_output(self, readings, weather_data, forecast_data):
        """Use Claude AI to analyze data and make exact predictions"""
        if not self.enabled:
            return self._fallback_prediction(readings, weather_data)
        
        # Create cache key
        cache_key = f"{len(readings)}_{weather_data.get('current', {}).get('clouds', 0)}"
        now = time.time()
        
        if (self.cache["data"] and 
            self.cache["key"] == cache_key and 
            (now - self.cache["ts"]) < self.cache_ttl):
            return self.cache["data"]
        
        # Prepare data for Claude
        recent_readings = readings[-50:] if len(readings) > 50 else readings
        
        # Calculate statistics
        ldr_values = [r['ldr'] for r in recent_readings]
        if not ldr_values:
            return self._fallback_prediction(readings, weather_data)
        
        avg_ldr = sum(ldr_values) / len(ldr_values)
        max_ldr = max(ldr_values)
        min_ldr = min(ldr_values)
        
        # Get current time info
        latest_ts = recent_readings[-1]['timestamp'] if recent_readings else int(time.time())
        dt = datetime.fromtimestamp(latest_ts)
        
        # Build prompt for Claude
        prompt = self._build_analysis_prompt(
            recent_readings, weather_data, forecast_data, 
            avg_ldr, max_ldr, min_ldr, dt
        )
        
        try:
            # Call Claude API
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2000,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data['content'][0]['text']
                
                # Parse Claude's response
                prediction = self._parse_ai_response(ai_response)
                
                # Cache the result
                self.cache = {
                    "ts": now,
                    "data": prediction,
                    "key": cache_key
                }
                
                # Store prediction in DB
                self._store_prediction(prediction, ai_response)
                
                return prediction
            else:
                print(f"Claude API error: {response.status_code}")
                return self._fallback_prediction(readings, weather_data)
                
        except Exception as e:
            print(f"Claude AI error: {e}")
            return self._fallback_prediction(readings, weather_data)
    
    def _build_analysis_prompt(self, readings, weather, forecast, avg_ldr, max_ldr, min_ldr, dt):
        """Build comprehensive prompt for Claude AI"""
        
        # Format recent readings
        readings_summary = []
        for i, r in enumerate(readings[-10:]):
            ts = datetime.fromtimestamp(r['timestamp'])
            readings_summary.append(f"  {ts.strftime('%H:%M:%S')} - LDR: {r['ldr']}/1023")
        
        weather_info = "Not available"
        if weather and 'current' in weather:
            weather_info = f"Temperature: {weather['current'].get('temp', 'N/A')}°C, Clouds: {weather['current'].get('clouds', 'N/A')}%"
        
        forecast_summary = "Not available"
        if forecast and isinstance(forecast, list) and len(forecast) > 0:
            forecast_summary = f"Next day: {forecast[0].get('predicted_sun_hours', 0)} hours, Clouds: {forecast[0].get('clouds', 0)}%"
        
        prompt = f"""You are an expert solar energy analyst. Analyze this solar panel system data and provide EXACT predictions.

CURRENT SYSTEM DATA:
==================
Current Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}
Hour of Day: {dt.hour}
Day of Week: {dt.strftime('%A')}

SENSOR READINGS (LDR values 0-1023, higher = more sunlight):
Latest 10 readings:
{chr(10).join(readings_summary)}

Statistics:
- Average LDR: {avg_ldr:.1f}/1023 ({(avg_ldr/1023*100):.1f}%)
- Maximum LDR: {max_ldr}/1023
- Minimum LDR: {min_ldr}/1023
- Total readings analyzed: {len(readings)}

CURRENT WEATHER:
{weather_info}

FORECAST:
{forecast_summary}

ANALYSIS REQUIRED:
==================
Based on the EXACT data above, provide your analysis in this JSON format:

{{
  "predicted_sun_hours": <number between 0-8>,
  "confidence": <percentage 0-100>,
  "reasoning": "<detailed explanation of your prediction>",
  "recommended_window": {{
    "start": "HH:MM",
    "end": "HH:MM"
  }},
  "load_recommendations": {{
    "heavy_loads": "<when to run: washer, EV, heater>",
    "medium_loads": "<when to run: iron, microwave>",
    "light_loads": "<when to run: lights, fans>"
  }},
  "energy_strategy": "<grid vs solar usage strategy>",
  "risk_factors": ["<list any factors that could affect prediction>"]
}}

IMPORTANT:
- Base predictions on ACTUAL sensor trends and patterns
- Consider time of day and weather conditions
- Be precise with time windows
- Explain your reasoning clearly
- Account for cloud coverage and weather changes
- Provide actionable recommendations

Respond ONLY with valid JSON, no additional text."""

        return prompt
    
    def _parse_ai_response(self, ai_response):
        """Parse Claude's JSON response into structured prediction"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = ai_response.strip()
            if json_str.startswith('```'):
                lines = json_str.split('\n')
                json_str = '\n'.join(lines[1:-1]) if len(lines) > 2 else json_str
                json_str = json_str.replace('```json', '').replace('```', '').strip()
            
            prediction = json.loads(json_str)
            
            # Validate and structure response
            return {
                "predicted_sun_hours": float(prediction.get("predicted_sun_hours", 0)),
                "confidence": int(prediction.get("confidence", 50)),
                "reasoning": prediction.get("reasoning", "AI analysis completed"),
                "recommended_window": prediction.get("recommended_window", {"start": "11:00", "end": "15:00"}),
                "load_recommendations": prediction.get("load_recommendations", {}),
                "energy_strategy": prediction.get("energy_strategy", ""),
                "risk_factors": prediction.get("risk_factors", []),
                "ai_powered": True,
                "source": "claude-ai"
            }
        except Exception as e:
            print(f"Parse error: {e}")
            print(f"AI Response: {ai_response[:200]}")
            return {
                "predicted_sun_hours": 0,
                "confidence": 0,
                "reasoning": "Failed to parse AI response",
                "recommended_window": {"start": "--", "end": "--"},
                "ai_powered": False,
                "error": str(e)
            }
    
    def _fallback_prediction(self, readings, weather_data):
        """Simple fallback when AI is unavailable"""
        if not readings:
            return {
                "predicted_sun_hours": 0,
                "confidence": 0,
                "reasoning": "No sensor data available",
                "recommended_window": {"start": "--", "end": "--"},
                "ai_powered": False
            }
        
        recent = readings[-20:] if len(readings) > 20 else readings
        avg_ldr = sum(r['ldr'] for r in recent) / len(recent)
        normalized = avg_ldr / 1023.0
        predicted_hours = round(normalized * 6.0, 2)
        
        if weather_data and 'current' in weather_data:
            clouds = weather_data['current'].get('clouds', 0)
            predicted_hours = round(predicted_hours * max(0.1, (1.0 - clouds / 150.0)), 2)
        
        if predicted_hours > 3.5:
            window = {"start": "11:30", "end": "15:00"}
            reason = "High sunlight detected"
        elif predicted_hours > 1.5:
            window = {"start": "09:00", "end": "11:30"}
            reason = "Moderate sunlight detected"
        else:
            window = {"start": "18:00", "end": "21:00"}
            reason = "Low sunlight detected"
        
        return {
            "predicted_sun_hours": predicted_hours,
            "confidence": 50,
            "reasoning": reason,
            "recommended_window": window,
            "ai_powered": False,
            "source": "fallback"
        }
    
    def _store_prediction(self, prediction, raw_response):
        """Store AI prediction in database for learning"""
        try:
            conn = sqlite3.connect(DB)
            c = conn.cursor()
            c.execute('''INSERT INTO ai_predictions (ts, prediction_data, actual_data) 
                         VALUES (?, ?, ?)''',
                      (int(time.time()), json.dumps(prediction), raw_response[:1000]))
            conn.commit()
            conn.close()
        except:
            pass

# Initialize Claude AI Engine
claude_engine = ClaudeAIPredictionEngine(ANTHROPIC_API_KEY)

# ------------------------------------------------------------
# WEATHER FUNCTIONS
# ------------------------------------------------------------
_weather_cache = {"ts": 0, "data": None}

def fetch_weather():
    if not OWM_KEY:
        return None
    
    now = time.time()
    if _weather_cache["data"] and (now - _weather_cache["ts"] < WEATHER_CACHE_TTL):
        return _weather_cache["data"]
    
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={LOCATION_LAT}&lon={LOCATION_LON}&appid={OWM_KEY}&units=metric")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        j = r.json()
        data = {
            "current": {
                "temp": j.get("main", {}).get("temp"),
                "clouds": j.get("clouds", {}).get("all", 0),
                "humidity": j.get("main", {}).get("humidity", 0),
                "wind_speed": j.get("wind", {}).get("speed", 0)
            }
        }
        _weather_cache["data"] = data
        _weather_cache["ts"] = now
        return data
    except Exception as e:
        print(f"Weather error: {e}")
        return None

_forecast_cache = {"ts": 0, "data": None}

def fetch_5day_forecast():
    if not OWM_KEY:
        return None
    
    now = time.time()
    if _forecast_cache["data"] and (now - _forecast_cache["ts"] < 600):
        return _forecast_cache["data"]
    
    try:
        url = (f"https://api.openweathermap.org/data/2.5/forecast"
               f"?lat={LOCATION_LAT}&lon={LOCATION_LON}&appid={OWM_KEY}&units=metric")
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        _forecast_cache["data"] = data
        _forecast_cache["ts"] = now
        return data
    except Exception as e:
        print(f"Forecast error: {e}")
        return None

def predict_from_3hr_forecast(forecast_json, days=5):
    """Convert forecast to daily predictions - enhanced for Claude"""
    if not forecast_json or "list" not in forecast_json:
        return []
    
    tz_offset = forecast_json.get("city", {}).get("timezone", 0)
    items = forecast_json["list"]
    
    days_map = {}
    for it in items:
        dt = int(it.get("dt", 0))
        local_ts = dt + int(tz_offset)
        dt_local = datetime.utcfromtimestamp(local_ts)
        date_str = dt_local.date().isoformat()
        hour = dt_local.hour
        clouds = it.get("clouds", {}).get("all", 0)
        pop = it.get("pop", 0.0)
        temp = it.get("main", {}).get("temp", 0)
        is_day = (6 <= hour <= 18)
        days_map.setdefault(date_str, []).append({
            "hour": hour, "is_day": is_day, "clouds": clouds, "pop": pop, "temp": temp
        })
    
    dates = sorted(days_map.keys())[:days]
    out = []
    
    for date_str in dates:
        slots = days_map.get(date_str, [])
        if not slots:
            out.append({
                "date": date_str,
                "predicted_sun_hours": 0.0,
                "clouds": 0,
                "pop": 0,
                "temp": 0,
                "recommended_window": {"start":"--","end":"--"},
                "suggestion": "No forecast data"
            })
            continue
        
        day_slots = [s for s in slots if s["is_day"]] or slots
        daylight_h = len(day_slots) * 3.0
        avg_cloud = sum(s["clouds"] for s in day_slots) / max(1, len(day_slots))
        avg_pop = sum(s["pop"] for s in day_slots) / max(1, len(day_slots))
        avg_temp = sum(s["temp"] for s in day_slots) / max(1, len(day_slots))
        
        predicted_hours = round(daylight_h * max(0.0, (1.0 - avg_cloud/100.0)) * max(0.0, (1.0 - avg_pop)), 2)
        
        if predicted_hours >= 4.0:
            window = {"start":"11:30","end":"15:00"}
            suggestion = "Heavy loads OK — run washer, EV charging, dishwasher, water heater."
        elif predicted_hours >= 1.5:
            window = {"start":"09:00","end":"11:30"}
            suggestion = "Moderate sunlight — run medium loads (iron, short heater runs)."
        else:
            window = {"start":"18:00","end":"21:00"}
            suggestion = "Low sunlight — avoid heavy loads; use grid or run light loads only."
        
        out.append({
            "date": date_str,
            "predicted_sun_hours": predicted_hours,
            "clouds": round(avg_cloud, 1),
            "pop": round(avg_pop, 2),
            "temp": round(avg_temp, 1),
            "recommended_window": window,
            "suggestion": suggestion
        })
    
    return out

# ------------------------------------------------------------
# DATABASE FUNCTIONS
# ------------------------------------------------------------
def insert_reading(deviceId, ldr, ts=None):
    if ts is None:
        ts = int(time.time())
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('INSERT INTO readings (deviceId, ts, ldr) VALUES (?, ?, ?)',
              (deviceId, ts, int(ldr)))
    conn.commit()
    conn.close()

def get_latest():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('SELECT deviceId, ts, ldr FROM readings ORDER BY id DESC LIMIT 1')
    row = c.fetchone()
    conn.close()
    if row:
        return {"deviceId": row[0], "timestamp": row[1], "ldr": row[2]}
    return None

def get_history(limit=200):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('SELECT deviceId, ts, ldr FROM readings ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    rows = rows[::-1]
    return [{"deviceId": r[0], "timestamp": r[1], "ldr": r[2]} for r in rows]

# ------------------------------------------------------------
# CLAUDE AI-POWERED RECOMMENDATION
# ------------------------------------------------------------
def compute_recommendation(latest):
    """Use Claude AI to compute exact recommendations"""
    if latest is None:
        return {
            "predicted_sun_hours": 0.0,
            "confidence": 0,
            "recommended_window": {"start":"--","end":"--"},
            "reason": "no-data",
            "ai_powered": False
        }
    
    # Get historical data
    history = get_history(limit=200)
    
    # Get weather and forecast
    weather = fetch_weather()
    forecast_data = fetch_5day_forecast()
    forecast_parsed = predict_from_3hr_forecast(forecast_data) if forecast_data else []
    
    # Use Claude AI for prediction
    prediction = claude_engine.predict_solar_output(history, weather, forecast_parsed)
    
    return prediction

# ------------------------------------------------------------
# FLASK ROUTES
# ------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sensor', methods=['POST'])
def api_sensor():
    data = request.get_json(force=True)
    deviceId = data.get('deviceId', 'esp1')
    ldr = data.get('ldr', None)
    ts = data.get('ts', None)
    if ldr is None:
        return jsonify({"error": "missing ldr"}), 400
    try:
        if ts:
            insert_reading(deviceId, int(ldr), int(ts))
        else:
            insert_reading(deviceId, int(ldr))
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        print(f"Insert error: {e}")
        return jsonify({"error": "server error"}), 500

@app.route('/api/latest', methods=['GET'])
def api_latest():
    """Get latest reading with Claude AI predictions"""
    latest = get_latest()
    if not latest:
        return jsonify({"status": "no-data"})
    
    rec = compute_recommendation(latest)
    result = dict(latest)
    result.update(rec)
    result["time_iso"] = datetime.utcfromtimestamp(result["timestamp"]).isoformat() + "Z"
    
    return jsonify(result)

@app.route('/api/history', methods=['GET'])
def api_history():
    limit = int(request.args.get('limit', 200))
    return jsonify(get_history(limit=limit))

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "bad payload"}), 400
    
    if "batch" in data:
        for item in data["batch"]:
            insert_reading(item.get("deviceId", "esp1"),
                           item.get("ldr", 0),
                           item.get("ts", None))
        return jsonify({"status": "ok", "inserted": len(data["batch"])})
    
    pattern = data.get("pattern", "sine")
    count = int(data.get("count", 60))
    delay = float(data.get("delay", 0))
    
    def run_sim():
        import math, random
        for i in range(count):
            x = i / max(1, count - 1)
            val = (0.5 - 0.5 * math.cos(x * math.pi * 2)) if pattern == "sine" else random.random()
            ldr = int(max(0, min(1023, val * 1023 + (random.random() - 0.5) * 120)))
            insert_reading("sim", ldr)
            if delay > 0:
                time.sleep(delay)
    
    threading.Thread(target=run_sim, daemon=True).start()
    return jsonify({"status": "started", "pattern": pattern, "count": count}), 200

@app.route('/api/weather', methods=['GET'])
def api_weather():
    w = fetch_weather()
    if not w:
        return jsonify({"status": "no-weather"}), 200
    current = w.get("current", {})
    return jsonify({
        "current": current,
        "next6": []
    })

@app.route('/api/forecast', methods=['GET'])
def api_forecast():
    days = int(request.args.get('days', 5))
    fjson = fetch_5day_forecast()
    if not fjson:
        return jsonify({"status":"no-weather"}), 200
    days_out = predict_from_3hr_forecast(fjson, days=days)
    return jsonify({"status":"ok", "days": days_out})

@app.route('/api/ai-status', methods=['GET'])
def api_ai_status():
    """Check AI engine status"""
    return jsonify({
        "claude_ai_enabled": claude_engine.enabled,
        "api_key_configured": bool(ANTHROPIC_API_KEY),
        "cache_active": claude_engine.cache["data"] is not None,
        "weather_api_configured": bool(OWM_KEY)
    })

# ------------------------------------------------------------
# RUN SERVER
# ------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("🤖 CLAUDE AI-POWERED SOLAR SCHEDULER")
    print("=" * 60)
    print(f"🧠 Claude AI: {'✅ ENABLED' if claude_engine.enabled else '❌ DISABLED (Set ANTHROPIC_API_KEY)'}")
    print(f"🌤️  Weather API: {'✅ Connected' if OWM_KEY else '❌ Not configured'}")
    print(f"🚀 Server: http://localhost:{PORT}")
    print("=" * 60)
    print("\n💡 This system uses Claude AI to analyze your solar data")
    print("   and provide EXACT predictions based on real patterns.\n")
    app.run(host='0.0.0.0', port=PORT, debug=True)