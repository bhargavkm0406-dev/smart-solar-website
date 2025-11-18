import requests
import json

GEMINI_API_KEY = "AIzaSyA2muMMHOhhZif7Sb29sKjQo_KZwlHFt3s"

print("🧪 Testing Gemini API...")
print("="*60)

# Test different API endpoints
endpoints = [
    f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
    f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}",
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
]

test_prompt = {
    "contents": [{
        "parts": [{"text": "Say 'Hello, I am working!' if you can read this."}]
    }]
}

for i, url in enumerate(endpoints, 1):
    print(f"\n🔍 Test {i}: {url.split('/models/')[1].split(':')[0]}")
    try:
        response = requests.post(url, json=test_prompt, timeout=10)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            text = result['candidates'][0]['content']['parts'][0]['text']
            print(f"   ✅ SUCCESS!")
            print(f"   Response: {text[:100]}")
            print(f"\n   🎉 THIS ENDPOINT WORKS! Use this in your app.py:")
            print(f"   {url.split('?')[0]}")
            break
        else:
            print(f"   ❌ Failed: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*60)