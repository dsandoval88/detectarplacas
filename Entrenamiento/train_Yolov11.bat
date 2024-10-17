@echo off
REM Ruta donde se encuentra tu archivo de datos y otros archivos necesarios
SET DATA_PATH=D:/TestYolo/entrenamiento/data
SET RUNS_PATH=D:/TestYolo/entrenamiento/ultralytics/runs
SET WEIGHTS_PATH=yolo11n.pt
REM Comando para ejecutar el contenedor Docker con YOLOv8 en una sola línea

docker run --ipc=host -it ^
    -v %RUNS_PATH%:/ultralytics/runs ^
    -v %DATA_PATH%:/usr/src/app/info ^
    ultralytics/ultralytics:latest ^
    yolo task=detect mode=train model=%WEIGHTS_PATH% ^
    data=/usr/src/app/info/data.yaml epochs=15 imgsz=640 batch=8


pause
