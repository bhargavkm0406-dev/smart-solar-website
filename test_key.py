# test_key.py - checks OpenWeather key and prints the API response status + start of text
import os, requests
k = os.getenv("OPENWEATHER_API_KEY")
print("OPENWEATHER_API_KEY present:", bool(k))
print("Key length:", len(k) if k else 0)
url = f"https://api.openweathermap.org/data/2.5/onecall?lat=13.05565&lon=77.50561&exclude=minutely,alerts&appid={k}&units=metric"
r = requests.get(url, timeout=10)
print("status_code:", r.status_code)
print("first 500 chars of response:\n")
print(r.text[:500])
