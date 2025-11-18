@echo off
   echo ================================================
   echo   AI SOLAR ENERGY SCHEDULER
   echo ================================================
   echo.
   echo Setting up environment variables...
   set GEMINI_API_KEY=AIzaSyA2muMMHOhhZif7Sb29sKjQo_KZwlHFt3s
   set OPENWEATHER_API_KEY=266cbcfc14167cde4293c8c572d95c62
   set LAT=13.05565
   set LON=77.50561
   
   echo Starting server...
   echo.
   python app.py
   pause
```
3. Save as `start.bat` in your `C:\solar-scheduler` folder

---

## 📂 **STEP 7: Verify Your Folder Structure**

Your folder should look like this:
```
C:\solar-scheduler\
├── app.py                  (Backend Python file)
├── start.bat               (Startup script)
├── templates\
│   └── index.html         (Frontend HTML file)
└── solar.db               (Will be created automatically)