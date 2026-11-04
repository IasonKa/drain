@echo off
rem One-time set-up of the Kratos drain CFD (Visual C++ runtime, Python 3.12, Kratos, gmsh, pyvista).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
pause
