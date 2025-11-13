#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <ArduinoJson.h>

const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASS";
const char* serverUrl = "http://YOUR_SERVER_IP:5000/api/sensor";

const int LDR_PIN = A0;
const char* deviceId = "esp1";

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400); Serial.print(".");
  }
  Serial.println("\nConnected");
}

void loop() {
  int ldrValue = analogRead(LDR_PIN);
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");
    StaticJsonDocument<200> doc;
    doc["deviceId"] = deviceId;
    doc["ts"] = millis()/1000;
    doc["ldr"] = ldrValue;
    String payload; serializeJson(doc, payload);
    int httpCode = http.POST(payload);
    if (httpCode > 0) {
      Serial.println("Posted: " + String(httpCode));
    } else {
      Serial.println("POST failed");
    }
    http.end();
  }
  delay(60000);
}
