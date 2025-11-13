import requests, time, random

SERVER = "http://localhost:5000/api/simulate"

payload = {"pattern": "sine", "count": 60, "delay": 0}
r = requests.post(SERVER, json=payload)

print("simulate POST ->", r.status_code, r.text)
