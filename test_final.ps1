Write-Host "=== FINAL TEST OF FIXED API ===" -ForegroundColor Cyan

# Test 1: Check if API health works
Write-Host "`n1. Testing /api/health endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod "http://localhost:5000/api/health" -Method GET
    Write-Host "   ✅ Health check PASSED" -ForegroundColor Green
    Write-Host "   Status: $($health.status)" -ForegroundColor Gray
} catch {
    Write-Host "   ❌ Health check FAILED: $_" -ForegroundColor Red
}

# Test 2: Test upload with minimal data
Write-Host "`n2. Testing upload with minimal data (ldr only)..." -ForegroundColor Yellow
$body1 = '{"ldr":500}'
try {
    $response1 = Invoke-RestMethod "http://localhost:5000/api/data/upload" -Method Post -Headers @{"Content-Type"="application/json";"X-API-Key"="solar2025"} -Body $body1
    Write-Host "   ✅ Upload SUCCESSFUL!" -ForegroundColor Green
    Write-Host "   Response: $($response1 | ConvertTo-Json -Compress)" -ForegroundColor Gray
} catch {
    Write-Host "   ❌ Upload FAILED: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $statusCode = $_.Exception.Response.StatusCode.value__
        Write-Host "   HTTP Status: $statusCode" -ForegroundColor Red
    }
}

# Test 3: Test upload with full data
Write-Host "`n3. Testing upload with full data..." -ForegroundColor Yellow
$body2 = @{
    ldr = 650
    deviceId = "esp8266-test"
    temperature = 27.5
    humidity = 55
    rssi = -65
    timestamp = [math]::Round((Get-Date).ToUnixTimeSeconds())
} | ConvertTo-Json

try {
    $response2 = Invoke-RestMethod "http://localhost:5000/api/data/upload" -Method Post -Headers @{"Content-Type"="application/json";"X-API-Key"="solar2025"} -Body $body2
    Write-Host "   ✅ Full upload SUCCESSFUL!" -ForegroundColor Green
    Write-Host "   Data ID: $($response2.data.id)" -ForegroundColor Gray
} catch {
    Write-Host "   ❌ Full upload FAILED: $_" -ForegroundColor Red
}

# Test 4: Verify data was stored
Write-Host "`n4. Verifying data was stored..." -ForegroundColor Yellow
try {
    $readings = Invoke-RestMethod "http://localhost:5000/api/data/readings?limit=3" -Method GET
    Write-Host "   ✅ Latest readings retrieved" -ForegroundColor Green
    Write-Host "   Count: $($readings.count)" -ForegroundColor Gray
    $readings.readings[0] | Format-List | Out-Host
} catch {
    Write-Host "   ❌ Could not retrieve readings: $_" -ForegroundColor Red
}

Write-Host "`n=== TEST COMPLETE ===" -ForegroundColor Cyan