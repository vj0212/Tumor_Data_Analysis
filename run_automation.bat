@echo off
REM Change to script directory
cd /d "C:\Users\jhavi\OneDrive\Desktop"

REM Set log file path
set LOG_FILE="C:\Users\jhavi\OneDrive\Desktop\automation_log.txt"

REM Add timestamp to log
echo. >> %LOG_FILE%
echo ====== RUN STARTED at %date% %time% ====== >> %LOG_FILE%

REM Run the Python script and append both stdout and stderr to the log
"C:\Users\jhavi\AppData\Local\Programs\Python\Python313\python.exe" "C:\Users\jhavi\OneDrive\Desktop\Automation.py" >> %LOG_FILE% 2>&1

echo ====== RUN COMPLETED at %date% %time% ====== >> %LOG_FILE%
echo. >> %LOG_FILE%
