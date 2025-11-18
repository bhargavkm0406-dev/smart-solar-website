import os
import time
import sqlite3
import json
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests

# ============================================================
# CONFIGURATION
# ============================================================
DB = 'solar.db'
PORT = int(os.getenv("PORT", 5000))

# API Keys
GEMINI_API_KEY = "AIzaSyA2muMMHOhhZif7Sb29sKjQo_KZwlHFt3s"
OWM_KEY = "266cbcfc14167cde4293c8c572d95c62"
LOCATION_LAT = "13.05565"
LOCATION_LON = "77.50561"

WEATHER_CACHE_TTL = 10 * 60
weather_cache = {"data": None, "timestamp": 0}

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
    conn.close()

# ============================================================
# GEMINI AI PREDICTION
# ============================================================
def get_gemini_prediction(ldr_value, history_data, weather_data):
    """Use Google Gemini to predict solar performance"""
    print(f"\n🔍 DEBUG: Gemini API Key present: {bool(GEMINI_API_KEY)}")
    print(f"🔍 DEBUG: API Key (first 20 chars): {GEMINI_API_KEY[:20] if GEMINI_API_KEY else 'None'}")
    
    if not GEMINI_API_KEY:
        return None
    
    try:
        print("📡 Calling Gemini API...")
        
        avg_ldr = sum([r[2] for r in history_data]) / len(history_data) if history_data else ldr_value
        
        prompt = f"""Analyze this solar panel data and provide JSON recommendations.

Current sensor: {ldr_value}/1023 (LDR)
Average (last hour): {avg_ldr:.1f}/1023
Time: {datetime.now().strftime('%H:%M')}
Weather: {weather_data.get('description', 'Unknown')}

Respond ONLY with this JSON (no markdown, no extra text):
{{
  "predicted_sun_hours": <number>,
  "confidence": <0-100>,
  "recommended_window": {{"start": "HH:MM", "end": "HH:MM"}},
  "reasoning": "<brief explanation>",
  "energy_strategy": "<actionable advice>"
}}"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
        
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 500,
                "response_mime_type": "application/json"
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=15)
        
        print(f"✅ Gemini Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Got response from Gemini!")
            
            # Debug: Show response structure
            print(f"📄 Response keys: {list(result.keys())}")
            if 'candidates' in result and len(result['candidates']) > 0:
                print(f"📄 Candidate keys: {list(result['candidates'][0].keys())}")
            
            # Extract text - try multiple formats
            text = None
            try:
                # Standard format
                text = result['candidates'][0]['content']['parts'][0]['text']
                print("✅ Used standard format")
            except (KeyError, IndexError, TypeError):
                try:
                    # Alternative: direct content text
                    text = result['candidates'][0]['content']['text']
                    print("✅ Used direct content.text format")
                except (KeyError, IndexError, TypeError):
                    try:
                        # Alternative: direct text
                        text = result['candidates'][0]['text']
                        print("✅ Used direct text format")
                    except (KeyError, IndexError, TypeError):
                        print(f"❌ Could not parse response: {json.dumps(result, indent=2)[:500]}")
                        return None
            
            if not text:
                print("❌ No text extracted")
                return None
            
            print(f"📄 Raw text preview: {text[:200]}...")
            
            # Clean and parse JSON
            text = text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]
            
            text = text.strip()
            prediction = json.loads(text)
            print(f"✅ Gemini prediction SUCCESS!")
            return prediction
            
        else:
            print(f"❌ Gemini error {response.status_code}: {response.text[:200]}")
            return None
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print(f"   Text was: {text[:200] if text else 'None'}")
        return None
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# WEATHER
# ============================================================
def fetch_weather():
    global weather_cache
    now = time.time()
    if weather_cache["data"] and (now - weather_cache["timestamp"]) < WEATHER_CACHE_TTL:
        return weather_cache["data"]
    
    if not OWM_KEY:
        return {"description": "Unknown", "temp": 25}
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={LOCATION_LAT}&lon={LOCATION_LON}&appid={OWM_KEY}&units=metric"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            weather_cache["data"] = {
                "description": data["weather"][0]["description"],
                "temp": data["main"]["temp"]
            }
            weather_cache["timestamp"] = now
            return weather_cache["data"]
    except:
        pass
    
    return {"description": "Unknown", "temp": 25}

# ============================================================
# FALLBACK
# ============================================================
def fallback_prediction(ldr_value):
    intensity = (ldr_value / 1023) * 100
    
    if intensity > 80:
        sun_hours = 6 + (intensity - 80) / 5
        window = {"start": "10:00", "end": "16:00"}
        reasoning = "Peak solar intensity detected"
    elif intensity > 50:
        sun_hours = 4 + (intensity - 50) / 10
        window = {"start": "11:00", "end": "15:00"}
        reasoning = "Good solar conditions"
    elif intensity > 20:
        sun_hours = 1 + (intensity - 20) / 15
        window = {"start": "12:00", "end": "14:00"}
        reasoning = "Moderate sunlight detected"
    else:
        sun_hours = intensity / 20
        window = {"start": "18:00", "end": "21:00"}
        reasoning = "Low sunlight detected"
    
    return {
        "predicted_sun_hours": round(sun_hours, 2),
        "confidence": 50,
        "recommended_window": window,
        "reasoning": reasoning,
        "energy_strategy": "Use grid power for heavy loads"
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
    
    return jsonify({"status": "ok", "timestamp": ts})

@app.route('/api/latest')
def latest():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 1")
    row = c.fetchone()
    
    if not row:
        conn.close()
        return jsonify({"status": "no-data"})
    
    c.execute("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 60")
    history = c.fetchall()
    conn.close()
    
    ldr_value = row[2]
    weather = fetch_weather()
    
    # Try Gemini
    prediction = get_gemini_prediction(ldr_value, history, weather)
    
    if prediction:
        response = {
            "timestamp": row[1],
            "time_iso": datetime.fromtimestamp(row[1], timezone.utc).isoformat().replace('+00:00', 'Z'),
            "ldr": ldr_value,
            "deviceId": row[3],
            "ai_powered": True,
            "source": "google-gemini-free",
            **prediction
        }
    else:
        prediction = fallback_prediction(ldr_value)
        response = {
            "timestamp": row[1],
            "time_iso": datetime.fromtimestamp(row[1], timezone.utc).isoformat().replace('+00:00', 'Z'),
            "ldr": ldr_value,
            "deviceId": row[3],
            "ai_powered": False,
            "source": "fallback",
            **prediction
        }
    
    return jsonify(response)

@app.route('/api/history')
def history():
    limit = request.args.get('limit', 100, type=int)
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

@app.route('/api/forecast')
def forecast():
    weather = fetch_weather()
    days = []
    
    for i in range(7):
        date = datetime.now() + timedelta(days=i)
        days.append({
            "date": date.strftime("%Y-%m-%d"),
            "day": date.strftime("%A"),
            "predicted_sun_hours": 5.5,
            "weather": weather.get("description", "Unknown"),
            "optimal_window": {"start": "10:00", "end": "15:00"}
        })
    
    return jsonify({"forecast": days})

@app.route('/api/ai-status')
def ai_status():
    return jsonify({
        "claude_ai_enabled": bool(GEMINI_API_KEY),
        "api_key_configured": bool(GEMINI_API_KEY),
        "ai_type": "google-gemini-free" if GEMINI_API_KEY else "none",
        "model": "gemini-2.5-flash-preview",
        "weather_api_configured": bool(OWM_KEY)
    })

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    pattern = data.get('pattern', 'sine')
    count = data.get('count', 100)
    
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    
    base_ts = int(time.time()) - (count * 60)
    
    for i in range(count):
        ts = base_ts + (i * 60)
        
        if pattern == 'sine':
            hour = (ts // 3600) % 24
            if 6 <= hour <= 18:
                ldr = int(512 + 400 * abs(((hour - 12) / 6)))
            else:
                ldr = int(50 + (i % 30))
        else:
            ldr = int(100 + (i * 5) % 900)
        
        c.execute("INSERT INTO readings (timestamp, ldr, deviceId) VALUES (?, ?, ?)",
                  (ts, ldr, 'sim'))
    
    conn.commit()
    conn.close()
    
    return jsonify({"status": "started", "pattern": pattern, "count": count})

# ============================================================
# STARTUP
# ============================================================
if __name__ == '__main__':
    init_db()
    
    print("\n" + "="*60)
    print("🤖 GOOGLE GEMINI AI SOLAR SCHEDULER (100% FREE)")
    print("="*60)
    
    if GEMINI_API_KEY:
        print("🧠 Gemini AI: ✅ ENABLED")
        print("   Model: gemini-2.5-flash-preview")
        print("   Limit: 15 requests/minute")
    else:
        print("🧠 Gemini AI: ❌ DISABLED")
    
    if OWM_KEY:
        print("🌤️  Weather API: ✅ Connected")
    else:
        print("🌤️  Weather API: ❌ Not configured")
    
    print(f"🚀 Server: http://localhost:{PORT}")
    print("="*60)
    print("\n💡 Using Google Gemini AI (100% FREE)")
    print("   Get your FREE API key: https://aistudio.google.com/app/apikey\n")
    
    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)