/*
 * Solar Energy IoT Sensor - ESP8266 Version
 * 
 * This sketch reads solar intensity (LDR), temperature, and humidity,
 * then sends the data to your Flask backend via HTTP POST.
 * 
 * Hardware Requirements:
 * - ESP8266 NodeMCU or similar
 * - LDR (Light Dependent Resistor)
 * - 10kΩ resistor
 * - DHT22 temperature/humidity sensor
 * 
 * Connections:
 * LDR: One leg to 3.3V, other leg to A0 and to GND via 10kΩ resistor
 * DHT22: VCC to 3.3V, GND to GND, DATA to D4
 * 
 * Author: Solar Energy Management System
 * Date: 2025
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
const char* DEVICE_ID = "solar-sensor-001";

// Pin definitions
#define LDR_PIN A0        // Analog pin for LDR sensor
#define DHT_PIN D4        // Digital pin for DHT22 sensor
#define DHT_TYPE DHT22    // DHT sensor type
#define LED_PIN LED_BUILTIN  // Built-in LED for status indication

// Timing configuration
const unsigned long SEND_INTERVAL = 60000;  // Send data every 60 seconds
const unsigned long RECONNECT_DELAY = 5000; // Wait 5 seconds before reconnecting

// ============================================================
// GLOBAL OBJECTS
// ============================================================

DHT dht(DHT_PIN, DHT_TYPE);
WiFiClient wifiClient;
HTTPClient http;

// Timing variables
unsigned long lastSendTime = 0;
unsigned long lastBlinkTime = 0;
bool ledState = false;

// Statistics
unsigned long successCount = 0;
unsigned long errorCount = 0;

// ============================================================
// SETUP FUNCTION
// ============================================================

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  delay(100);
  Serial.println();
  Serial.println("====================================");
  Serial.println("  Solar Energy IoT Sensor v1.0");
  Serial.println("====================================");
  
  // Initialize pins
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);  // Turn on LED during setup
  
  // Initialize DHT sensor
  Serial.println("Initializing DHT22 sensor...");
  dht.begin();
  delay(2000);  // DHT sensor needs time to stabilize
  
  // Connect to WiFi
  connectWiFi();
  
  // Test sensors
  testSensors();
  
  // Initial data send
  Serial.println("\nSending initial data...");
  sendSensorData();
  
  digitalWrite(LED_PIN, LOW);  // Turn off LED
  Serial.println("\nSetup complete! Entering main loop...");
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
  
  // Blink LED to show activity
  if (millis() - lastBlinkTime >= 1000) {
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
    lastBlinkTime = millis();
  }
  
  // Send sensor data at specified interval
  unsigned long currentTime = millis();
  if (currentTime - lastSendTime >= SEND_INTERVAL) {
    sendSensorData();
    lastSendTime = currentTime;
    
    // Print statistics
    printStatistics();
  }
  
  // Small delay to prevent watchdog issues
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
    
    // Blink LED faster during connection
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
  // Read LDR value (0-1023)
  int ldrValue = analogRead(LDR_PIN);
  
  Serial.print("🔍 DEBUG - Raw ADC: ");
  Serial.println(ldrValue);
  
  // Apply smoothing by taking multiple readings
  int sum = ldrValue;
  for (int i = 1; i < 5; i++) {
    delay(10);
    int reading = analogRead(LDR_PIN);
    Serial.print("   Reading ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(reading);
    sum += reading;
  }
  
  int average = sum / 5;
  Serial.print("   Average: ");
  Serial.println(average);
  
  return average;  // Return average
}

float readTemperature() {
  float temp = dht.readTemperature();
  
  // Validate reading
  if (isnan(temp)) {
    Serial.println("⚠ Warning: DHT temperature reading failed!");
    return 25.0;  // Return default value
  }
  
  return temp;
}

float readHumidity() {
  float humidity = dht.readHumidity();
  
  // Validate reading
  if (isnan(humidity)) {
    Serial.println("⚠ Warning: DHT humidity reading failed!");
    return 60.0;  // Return default value
  }
  
  return humidity;
}

// ============================================================
// DATA TRANSMISSION
// ============================================================

void sendSensorData() {
  // Check WiFi status
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
  
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");
  
  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println(" %");
  
  Serial.print("WiFi Signal: ");
  Serial.print(rssi);
  Serial.println(" dBm");
  
  // Create JSON payload
  StaticJsonDocument<256> jsonDoc;
  jsonDoc["deviceId"] = DEVICE_ID;
  jsonDoc["ldr"] = ldrValue;
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
      Serial.println("⚠ Warning: Server returned non-200 status");
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

void testSensors() {
  Serial.println("\n=== Sensor Test ===");
  
  // Test LDR
  Serial.print("Testing LDR... ");
  int ldr = readLDR();
  Serial.print("Value: ");
  Serial.println(ldr);
  
  // Test DHT22
  Serial.print("Testing DHT22... ");
  float temp = readTemperature();
  float hum = readHumidity();
  
  if (!isnan(temp) && !isnan(hum)) {
    Serial.println("OK");
    Serial.print("  Temperature: ");
    Serial.print(temp);
    Serial.println(" °C");
    Serial.print("  Humidity: ");
    Serial.print(hum);
    Serial.println(" %");
  } else {
    Serial.println("FAILED!");
  }
  
  Serial.println("==================\n");
}

void printStatistics() {
  unsigned long totalAttempts = successCount + errorCount;
  float successRate = 0;
  
  if (totalAttempts > 0) {
    successRate = (float)successCount / totalAttempts * 100.0;
  }
  
  Serial.println("\n=== Statistics ===");
  Serial.print("Total Transmissions: ");
  Serial.println(totalAttempts);
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
