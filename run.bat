@echo off
REM Activate venv and run app (Windows CMD)
if exist venv\Scripts\activate (
  call venv\Scripts\activate
) else (
  echo No virtual env found. Create one with: py -3.11 -m venv venv
  pause
  exit /b 1
)

python app_enhanced.py
pause
