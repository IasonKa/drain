@echo off
rem Mesh, run and picture both Kratos drain CFD cases on all CPU cores (takes many hours).
rem Options: RUN.bat -Cases rec_tilt10:8   |   RUN.bat -Threads 16   |   RUN.bat -Force   |   RUN.bat -FineUSZ
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" %*
pause
