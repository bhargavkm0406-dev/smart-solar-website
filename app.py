# app.py — Smart Solar Scheduler (final version with working weather API)
import os
import time
import sqlite3
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import requests

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DB = 'solar.db'
PORT = int(os.getenv("PORT", 5000))
OWM_KEY = os.getenv("OPENWEATHER_API_KEY")
LOCATION_LAT = os.getenv("LAT", "13.05565")
LOCATION_LON = os.getenv("LON", "77.50561")
WEATHER_CACHE_TTL = 10 * 60  # 10 minutes cache

# ------------------------------------------------------------
# FLASK APP INITIALIZATION
# ------------------------------------------------------------
app = Flask(__name__, instance_path=os.path.join(os.getcwd(), 'instance'))
CORS(app)

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
    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------
# WEATHER FETCH (Simple /weather endpoint)
# ------------------------------------------------------------
_weather_cache = {"ts": 0, "data": None}

def fetch_weather():
    """Fetch simple current weather info (temperature + cloud %)"""
    if not OWM_KEY:
        print("No OpenWeather API key set.")
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
                "clouds": j.get("clouds", {}).get("all", 0)
            },
            "hourly": []
        }
        _weather_cache["data"] = data
        _weather_cache["ts"] = now
        print("Weather updated:", data)
        return data
    except Exception as e:
        print("Weather fetch failed:", e)
        return None

# ------------------------------------------------------------
# SENSOR & DATABASE LOGIC
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

# ---------------------------
# Simple forecasting helpers (Holt's linear trend)
# ---------------------------
def holt_linear_forecast(values, n_forecast=6, alpha=0.6, beta=0.2):
    """
    values: list of recent numeric LDR readings (oldest -> newest)
    returns: list of n_forecast forecasted values
    Implements simple Holt's linear method (level + trend).
    """
    if not values:
        return [0.0] * n_forecast
    if len(values) == 1:
        return [values[0]] * n_forecast

    # initialize
    level = values[0]
    trend = values[1] - values[0]
    # smoothing
    for i in range(1, len(values)):
        val = values[i]
        prev_level = level
        level = alpha * val + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

    # forecast
    forecasts = []
    for k in range(1, n_forecast + 1):
        forecasts.append(max(0.0, level + trend * k))
    return forecasts


def predict_future_hours_from_history(history_readings, n_hours=6, clouds_pct=None):
    """
    history_readings: list of dicts like returned by get_history() (ordered oldest->newest)
    n_hours: number of hourly points to forecast (default 6)
    clouds_pct: optional cloud percentage (0-100) to adjust forecast
    Returns: (predicted_hours:float, forecasted_ldr:list)
    """
    vals = [min(1023, max(0, int(r.get("ldr", 0)))) for r in history_readings if r.get("ldr") is not None]
    if not vals:
        vals = [0]
    vals = vals[-60:]  # use up to 60 recent points

    # forecast next n_hours LDR values using Holt linear
    forecast_ldr = holt_linear_forecast(vals, n_forecast=n_hours, alpha=0.6, beta=0.2)

    avg_predicted_ldr = sum(forecast_ldr) / max(1, len(forecast_ldr))
    normalized = float(avg_predicted_ldr) / 1023.0
    predicted_hours = round(normalized * 6.0, 2)

    if clouds_pct is not None:
        predicted_hours = round(predicted_hours * max(0.0, (1.0 - clouds_pct / 150.0)), 2)

    return predicted_hours, forecast_ldr

# ------------------------------------------------------------
# SOLAR RECOMMENDATION LOGIC
# ------------------------------------------------------------
def compute_recommendation(latest):
    """
    Uses recent LDR history + simple Holt forecast + weather clouds to predict sunlight hours
    and compute recommended window.
    """
    if latest is None:
        return {"predicted_sun_hours": 0.0, "recommended_window": {"start":"--","end":"--"}, "reason":"no-data"}

    # Get recent history (oldest->newest)
    history = get_history(limit=120)  # use up to 120 recent samples (you can tune)
    # pass cloud% from current weather (if available)
    weather = fetch_weather()
    clouds_pct = None
    if weather and "current" in weather:
        clouds_pct = weather["current"].get("clouds", None)

    # Use the forecasting helper: predict next 6 hourly-equivalent LDR values
    predicted_hours, predicted_ldr_series = predict_future_hours_from_history(history, n_hours=6, clouds_pct=clouds_pct)

    # Build recommended window using thresholds (same style as before)
    if predicted_hours > 3.5:
        window = {"start":"11:30","end":"15:00"}
        reason = "High forecasted sunlight — midday recommended."
    elif predicted_hours > 1.5:
        window = {"start":"09:00","end":"11:30"}
        reason = "Moderate forecasted sunlight."
    else:
        window = {"start":"18:00","end":"21:00"}
        reason = "Low sunlight — use grid or avoid heavy loads."

    # You can optionally include the small forecast series in the returned dict
    return {
        "predicted_sun_hours": predicted_hours,
        "recommended_window": window,
        "reason": reason,
        "forecast_ldr": predicted_ldr_series  # optional: small series for debugging / UI
    }

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
        print("Insert error:", e)
        return jsonify({"error": "server error"}), 500

@app.route('/api/latest', methods=['GET'])
def api_latest():
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
        return jsonify({"status": "no-weather", "note": "Invalid or missing API key"}), 200
    current = w.get("current", {})
    return jsonify({
        "current": {"temp": current.get("temp"), "clouds": current.get("clouds")},
        "next6": []
    })

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# ------------------------------------------------------------
# RUN SERVER
# ------------------------------------------------------------
if __name__ == '__main__':
    print(f"Starting server on port {PORT}. Open http://localhost:{PORT} in your browser.")
    app.run(host='0.0.0.0', port=PORT, debug=True)
