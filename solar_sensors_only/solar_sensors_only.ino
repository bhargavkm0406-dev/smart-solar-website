/*
 * Solar Energy Monitoring System - Sensors Only
 * 
 * Features:
 * - DHT22 & DHT11 temperature/humidity sensors
 * - LDR light intensity sensing
 * - WiFi data transmission to Flask backend
 * - Real-time monitoring and ML predictions
 * 
 * Hardware:
 * - ESP8266 NodeMCU
 * - DHT22 on D4
 * - DHT11 on D5 (optional backup)
 * - LDR on A0 with 10kΩ resistor
 */

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <DHT.h>
#include <ArduinoJson.h>

// ============================================================
// CONFIGURATION - UPDATE THESE VALUES
// ============================================================

// WiFi credentials
const char* WIFI_SSID = "Rbk";
const char* WIFI_PASSWORD = "12345678";

// Server configuration
const char* SERVER_URL = "http://10.129.142.245:5000/api/data/upload";
const char* API_KEY = "solar2025";

// Device identification
const char* DEVICE_ID = "solar-monitor-001";

// Pin definitions
#define LDR_PIN A0           // LDR sensor (analog)
#define DHT22_PIN D4         // DHT22 sensor
#define DHT11_PIN D5         // DHT11 sensor (optional)
#define LED_PIN LED_BUILTIN  // Status LED

// DHT sensor types
#define DHT22_TYPE DHT22
#define DHT11_TYPE DHT11

// Timing configuration
const unsigned long SEND_INTERVAL = 60000;  // Send data every 60 seconds

// ============================================================
// GLOBAL OBJECTS
// ============================================================

DHT dht22(DHT22_PIN, DHT22_TYPE);
DHT dht11(DHT11_PIN, DHT11_TYPE);  // Optional backup sensor
WiFiClient wifiClient;
HTTPClient http;

// State variables
unsigned long lastSendTime = 0;
unsigned long successCount = 0;
unsigned long errorCount = 0;

// ============================================================
// SETUP FUNCTION
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(100);
  
  Serial.println();
  Serial.println("====================================");
  Serial.println("  Solar Monitoring System v1.0");
  Serial.println("====================================");
  
  // Initialize LED
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  
  // Initialize DHT sensors
  Serial.println("Initializing sensors...");
  dht22.begin();
  dht11.begin();
  delay(2000);  // DHT needs time to stabilize
  
  // Connect to WiFi
  connectWiFi();
  
  // Test all sensors
  testAllSensors();
  
  // Send initial data
  Serial.println("\nSending initial data...");
  sendSensorData();
  
  digitalWrite(LED_PIN, LOW);
  Serial.println("\nSetup complete! Monitoring started...");
  Serial.println("====================================\n");
}

// ============================================================
// MAIN LOOP
// ============================================================

void loop() {
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected! Reconnecting...");
    connectWiFi();
  }
  
  unsigned long currentTime = millis();
  
  // Send sensor data at interval
  if (currentTime - lastSendTime >= SEND_INTERVAL) {
    sendSensorData();
    lastSendTime = currentTime;
    printStatistics();
  }
  
  // Blink LED to show activity
  static unsigned long lastBlink = 0;
  if (currentTime - lastBlink >= 1000) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    lastBlink = currentTime;
  }
  
  delay(100);
}

// ============================================================
// WIFI CONNECTION
// ============================================================

void connectWiFi() {
  Serial.println("\nConnecting to WiFi...");
  Serial.print("SSID: ");
  Serial.println(WIFI_SSID);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓ WiFi connected successfully!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal Strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
    digitalWrite(LED_PIN, LOW);
  } else {
    Serial.println("\n✗ WiFi connection failed!");
    Serial.println("Continuing in offline mode...");
    digitalWrite(LED_PIN, HIGH);
  }
}

// ============================================================
// SENSOR READING FUNCTIONS
// ============================================================

int readLDR() {
  // Take multiple readings and average
  int sum = 0;
  for (int i = 0; i < 5; i++) {
    sum += analogRead(LDR_PIN);
    delay(10);
  }
  return sum / 5;
}

float readTemperature() {
  // Try DHT22 first
  float temp = dht22.readTemperature();
  
  // If DHT22 fails, try DHT11
  if (isnan(temp)) {
    Serial.println("⚠ DHT22 failed, using DHT11...");
    temp = dht11.readTemperature();
    
    // If both fail, return default
    if (isnan(temp)) {
      Serial.println("⚠ Both DHT sensors failed!");
      return 25.0;  // Default fallback
    }
  }
  
  return temp;
}

float readHumidity() {
  // Try DHT22 first
  float hum = dht22.readHumidity();
  
  // If DHT22 fails, try DHT11
  if (isnan(hum)) {
    Serial.println("⚠ DHT22 failed, using DHT11...");
    hum = dht11.readHumidity();
    
    // If both fail, return default
    if (isnan(hum)) {
      Serial.println("⚠ Both DHT sensors failed!");
      return 60.0;  // Default fallback
    }
  }
  
  return hum;
}

// ============================================================
// DATA TRANSMISSION
// ============================================================

void sendSensorData() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("✗ Cannot send data: WiFi not connected");
    errorCount++;
    return;
  }
  
  Serial.println("\n--- Sensor Reading ---");
  
  // Read all sensors
  int ldrValue = readLDR();
  float temperature = readTemperature();
  float humidity = readHumidity();
  int rssi = WiFi.RSSI();
  
  // Display readings
  Serial.print("LDR Value: ");
  Serial.print(ldrValue);
  Serial.println(" (0-1023)");
  
  // Calculate solar intensity percentage
  int solarPercent = map(ldrValue, 0, 1023, 0, 100);
  Serial.print("Solar Intensity: ");
  Serial.print(solarPercent);
  Serial.println("%");
  
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");
  
  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");
  
  Serial.print("WiFi Signal: ");
  Serial.print(rssi);
  Serial.println(" dBm");
  
  // Determine solar status
  String solarStatus;
  if (ldrValue >= 700) {
    solarStatus = "☀️ High Solar - Excellent for energy production";
  } else if (ldrValue >= 400) {
    solarStatus = "🌤️ Medium Solar - Good for light loads";
  } else if (ldrValue >= 200) {
    solarStatus = "⛅ Low Solar - Limited energy available";
  } else {
    solarStatus = "🌙 Night/Dark - No solar energy";
  }
  Serial.println(solarStatus);
  
  // Create JSON payload
  StaticJsonDocument<512> jsonDoc;
  jsonDoc["deviceId"] = DEVICE_ID;
  jsonDoc["ldr"] = ldrValue;
  jsonDoc["solarPercent"] = solarPercent;
  jsonDoc["temperature"] = round(temperature * 10) / 10.0;
  jsonDoc["humidity"] = round(humidity * 10) / 10.0;
  jsonDoc["timestamp"] = millis() / 1000;
  jsonDoc["rssi"] = rssi;
  
  String jsonPayload;
  serializeJson(jsonDoc, jsonPayload);
  
  Serial.println("\nSending to server...");
  Serial.print("URL: ");
  Serial.println(SERVER_URL);
  Serial.print("Payload: ");
  Serial.println(jsonPayload);
  
  // Send HTTP POST request
  http.begin(wifiClient, SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-API-Key", API_KEY);
  http.setTimeout(10000);  // 10 second timeout
  
  int httpResponseCode = http.POST(jsonPayload);
  
  // Handle response
  // Handle response
if (httpResponseCode > 0) {
  Serial.print("✓ HTTP Response Code: ");
  Serial.println(httpResponseCode);
  
  String response = http.getString();
  Serial.print("Server Response: ");
  Serial.println(response);
  
  if (httpResponseCode == 200) {
    successCount++;
    
    // Blink LED 3 times to indicate success
    for (int i = 0; i < 3; i++) {
      digitalWrite(LED_PIN, HIGH);
      delay(100);
      digitalWrite(LED_PIN, LOW);
      delay(100);
    }
  } else {
    errorCount++;
    Serial.print("⚠ Warning: Server returned non-200 status: ");
    Serial.println(httpResponseCode);
    Serial.print("Response: ");
    Serial.println(response);
  }
} else {
    Serial.print("✗ Error sending data. HTTP Code: ");
    Serial.println(httpResponseCode);
    Serial.print("Error: ");
    Serial.println(http.errorToString(httpResponseCode));
    errorCount++;
  }
  
  http.end();
  Serial.println("---------------------\n");
}

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

void testAllSensors() {
  Serial.println("\n=== Sensor Test ===");
  
  // Test LDR
  Serial.print("Testing LDR... ");
  int ldr = readLDR();
  Serial.print("Value: ");
  Serial.println(ldr);
  
  // Test DHT22
  Serial.print("Testing DHT22... ");
  float temp22 = dht22.readTemperature();
  float hum22 = dht22.readHumidity();
  
  if (!isnan(temp22) && !isnan(hum22)) {
    Serial.println("✓ OK");
    Serial.print("  Temperature: ");
    Serial.print(temp22);
    Serial.println(" °C");
    Serial.print("  Humidity: ");
    Serial.print(hum22);
    Serial.println(" %");
  } else {
    Serial.println("✗ FAILED");
  }
  
  // Test DHT11 (optional)
  Serial.print("Testing DHT11... ");
  float temp11 = dht11.readTemperature();
  float hum11 = dht11.readHumidity();
  
  if (!isnan(temp11) && !isnan(hum11)) {
    Serial.println("✓ OK");
    Serial.print("  Temperature: ");
    Serial.print(temp11);
    Serial.println(" °C");
    Serial.print("  Humidity: ");
    Serial.print(hum11);
    Serial.println(" %");
  } else {
    Serial.println("Not connected or failed");
  }
  
  Serial.println("==================\n");
}

void printStatistics() {
  unsigned long total = successCount + errorCount;
  float successRate = (total > 0) ? (float)successCount / total * 100.0 : 0;
  
  Serial.println("\n=== Statistics ===");
  Serial.print("Total Transmissions: ");
  Serial.println(total);
  Serial.print("Successful: ");
  Serial.println(successCount);
  Serial.print("Failed: ");
  Serial.println(errorCount);
  Serial.print("Success Rate: ");
  Serial.print(successRate);
  Serial.println("%");
  Serial.print("Uptime: ");
  Serial.print(millis() / 1000);
  Serial.println(" seconds");
  Serial.println("==================\n");
}

// ============================================================
// END OF FILE
// ============================================================
