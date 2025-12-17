import requests

url = "http://localhost:5000/api/data/upload"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "solar2025"
}

# Test 1: Minimal data
data1 = {"ldr": 500}
print("Test 1 - Minimal data:")
response = requests.post(url, headers=headers, json=data1)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}\n")

# Test 2: Full data
data2 = {
    "ldr": 500,
    "deviceId": "test-device",
    "temperature": 25.5,
    "humidity": 60,
    "rssi": -45
}
print("Test 2 - Full data:")
response = requests.post(url, headers=headers, json=data2)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")