Write-Host "=== FIXED API TEST ===" -ForegroundColor Cyan

# Fix PowerShell date conversion
function Get-UnixTimestamp {
    return [int][double]::Parse((Get-Date -UFormat %s))
}

# Test 1: Check if root works
Write-Host "`n1. Testing root endpoint (/)..." -ForegroundColor Yellow
try {
    $root = Invoke-WebRequest "http://localhost:5000/" -Method GET
    Write-Host "   ✅ Root endpoint works" -ForegroundColor Green
} catch {
    Write-Host "   ❌ Root endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: Check /health (not /api/health)
Write-Host "`n2. Testing /health endpoint..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod "http://localhost:5000/health" -Method GET
    Write-Host "   ✅ Health check PASSED" -ForegroundColor Green
    Write-Host "   Status: $($health.status)" -ForegroundColor Gray
} catch {
    Write-Host "   ❌ Health check FAILED: $_" -ForegroundColor Red
}

# Test 3: Test upload with minimal data (TEMPORARY FIX - bypass validation)
Write-Host "`n3. Testing upload with minimal data..." -ForegroundColor Yellow
$body1 = '{"ldr":500}'
try {
    $response1 = Invoke-RestMethod "http://localhost:5000/api/data/upload" -Method Post -Headers @{"Content-Type"="application/json"} -Body $body1
    Write-Host "   ⚠ Upload without API key: $($response1.message)" -ForegroundColor Yellow
} catch {
    Write-Host "   ✅ Correctly failed without API key" -ForegroundColor Green
}

# Test 4: Test upload WITH API key
Write-Host "`n4. Testing upload WITH API key..." -ForegroundColor Yellow
$body2 = '{"ldr":500}'
try {
    $response2 = Invoke-RestMethod "http://localhost:5000/api/data/upload" -Method Post -Headers @{"Content-Type"="application/json";"X-API-Key"="solar2025"} -Body $body2
    Write-Host "   ✅ Upload SUCCESSFUL!" -ForegroundColor Green
    Write-Host "   Response: $($response2 | ConvertTo-Json -Compress)" -ForegroundColor Gray
} catch {
    Write-Host "   ❌ Upload FAILED: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $reader.BaseStream.Position = 0
        $responseBody = $reader.ReadToEnd()
        Write-Host "   Error details: $responseBody" -ForegroundColor Red
    }
}

# Test 5: Verify data was stored
Write-Host "`n5. Verifying data was stored..." -ForegroundColor Yellow
try {
    $readings = Invoke-RestMethod "http://localhost:5000/api/data/readings?limit=1" -Method GET
    Write-Host "   ✅ Latest reading retrieved" -ForegroundColor Green
    Write-Host "   LDR value: $($readings.readings[0].ldr)" -ForegroundColor Gray
} catch {
    Write-Host "   ❌ Could not retrieve readings: $_" -ForegroundColor Red
}

Write-Host "`n=== TEST COMPLETE ===" -ForegroundColor Cyan