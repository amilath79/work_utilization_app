@echo off
cd /d "C:\forlogssystems\app"
echo %date% %time% - Starting model training >> logs\scheduler.log
py train_models2.py >> logs\training_output.log 2>&1
echo %date% %time% - Model training completed >> logs\scheduler.log