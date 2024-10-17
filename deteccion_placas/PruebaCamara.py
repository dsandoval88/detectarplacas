#pip install opencv-python
#pip install torch torchvision
#pip install ultralytics
#pip install easyocr

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO('plate_detection/plate_detection.pt')

# Iniciar la captura de video desde la cámara de la laptop
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Umbral de confianza (0.5 o más)
CONFIDENCE_THRESHOLD = 0.5

# Configurar matplotlib para no bloquear
plt.ion()  # Activar modo interactivo

# Crear una ventana de visualización
fig, ax = plt.subplots()

def show_frame_with_matplotlib(frame):
    # Convertir el frame de BGR a RGB para matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.clear()  # Limpiar la imagen previa
    ax.imshow(frame_rgb)
    ax.set_axis_off()  # Ocultar los ejes
    plt.draw()  # Dibujar la imagen
    plt.pause(0.001)  # Pausar brevemente para actualizar la imagen

# Bucle para capturar los frames y realizar la detección
while True:
    ret, frame = cap.read()  # Capturar frame por frame
    if not ret:
        print("No se pudo recibir el frame. Saliendo...")
        break

    # Hacer la detección en el frame
    results = model(frame)

    # Obtener las predicciones
    for result in results:
        for box in result.boxes:
            # Filtrar por confianza
            if box.conf >= CONFIDENCE_THRESHOLD:
                # Obtener coordenadas de la caja
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # Dibujar la caja alrededor de la placa detectada
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Añadir el texto 'Placa' en la detección
                cv2.putText(frame, 'Placa', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Mostrar el frame con matplotlib
    show_frame_with_matplotlib(frame)

    # Salir del bucle si se presiona la tecla 'q' (manual)
    if plt.waitforbuttonpress(0.1):  # Presiona 'q' para salir
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
plt.ioff()  # Desactivar el modo interactivo
plt.show()  # Mantener la última imagen en pantalla
