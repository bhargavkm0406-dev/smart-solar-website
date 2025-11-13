# -*- coding: utf-8 -*-
import os, requests

print("Testing OpenWeather API key...")
k = os.getenv("OPENWEATHER_API_KEY")
print("KEY present:", bool(k), "len:", len(k) if k else 0)

url = f"https://api.openweathermap.org/data/2.5/weather?lat=13.05565&lon=77.50561&appid={k}&units=metric"
r = requests.get(url, timeout=10)
print("status:", r.status_code)
print(r.text[:800])
