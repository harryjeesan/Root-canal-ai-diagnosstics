@echo off
echo ==========================================
echo   Fractured Teeth AI Diagnostic System
echo ==========================================
echo.
cd fractured_teeth_module
echo Running Optimized Binary R-CNN (Finer Heatmaps)...
..\venv_rcnn\Scripts\python.exe test_optimized_rcnn.py
echo.
echo Running Multi-Class R-CNN (Original ResNet50)...
..\venv_rcnn\Scripts\python.exe test_rcnn_pro.py
echo.
echo All tests complete! Check 'fractured_teeth_module' for results.
pause
