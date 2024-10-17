import tornado.ioloop
import tornado.web
import base64
import torch
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

class PlateDetection:
    def __init__(self, model, lectura):
        self.model = model
        self.lectura = lectura

    def plate_detection(self, imagen):
        results = self.model(imagen)  # Realizar la detección
        #print("results: ",results)
        detecciones = results[0].boxes.xyxy.cpu().numpy()  # Extraer las coordenadas de las cajas
        #print("detecciones: ",detecciones)
        confidences = results[0].boxes.conf.cpu().numpy()  # Extraer las confianzas de las detecciones
        #print("confidences: ",confidences)
        class_ids = results[0].boxes.cls.cpu().numpy()  # Extraer las clases de las detecciones
        #print("class_ids: ",class_ids)

        # Filtrar detecciones que superen el umbral de confianza y que sean de clase "placa"
        placas_detectadas = []
        for i, (x_min, y_min, x_max, y_max) in enumerate(detecciones):
            if confidences[i] > 0.1 and class_ids[i] == 0:  # Asumiendo que la clase 0 es "placa"
                placas_detectadas.append((imagen[int(y_min):int(y_max), int(x_min):int(x_max)], (x_min, y_min, x_max, y_max)))

        return placas_detectadas

    # def plate_detection(self, imagen):
    #     results = self.model(imagen)
    #     detecciones = results.pandas().xyxy[0]
    #     placas = detecciones[(detecciones['confidence'] > 0.5) & 
    #                          (detecciones['name'].str.contains('placa'))]
    #     if placas.empty:
    #         return []

    #     placas_detectadas = []
    #     for _, placa in placas.iterrows():
    #         x_min, y_min, x_max, y_max = int(placa['xmin']), int(placa['ymin']), int(placa['xmax']), int(placa['ymax'])
    #         placas_detectadas.append((imagen[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)))

    #     return placas_detectadas

    def filtros_morfologicos(self, imagen):
        kernel = np.ones((2, 2), np.uint8)
        imagen_erosionada = cv2.erode(imagen, kernel, iterations=1)
        filtro_promedio = cv2.filter2D(imagen_erosionada, -1, (1, 1))
        return filtro_promedio

    def rotar_imagen(self, imagen, grados):
        if grados == 90:
            return cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)
        elif grados == 180:
            return cv2.rotate(imagen, cv2.ROTATE_180)
        elif grados == 270:
            return cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return imagen
    
    def leer(self, imagen):
        placa_ocr = self.lectura.readtext(imagen)
        salida = []
        for a in placa_ocr:
            salida.append(a[-2])
        return salida

    def procesar_placas(self, placas_detectadas):
        resultados = []
        for placa, coordenadas in placas_detectadas:
            placa_filtrada = self.filtros_morfologicos(placa)
            texto_placa = self.leer(placa_filtrada)

            if not texto_placa:
                for grados in [90, 180, 270]:
                    placa_rotada = self.rotar_imagen(placa_filtrada, grados)
                    texto_placa = self.leer(placa_rotada)
                    if texto_placa:
                        break

            if not texto_placa:
                texto_placa = ['Lectura fallida']

            resultados.append({
                'placa': texto_placa,
                'coordenadas': coordenadas
            })
        return resultados

class DefaultRequestHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Detector de placas operativo")

class PhotoRequestHandler(tornado.web.RequestHandler):
    def initialize(self, detector):
        self.detector = detector

    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        foto_base64 = data.get('foto')
        tipo_imagen = data.get('tipo')
        prefix = f'data:image/{tipo_imagen};base64,'
        if foto_base64.startswith(prefix):
            foto_base64 = foto_base64.replace(prefix, '')
        imagen_bytes = base64.b64decode(foto_base64)
        imagen_np = np.frombuffer(imagen_bytes, np.uint8)
        imagen = cv2.imdecode(imagen_np, cv2.IMREAD_COLOR)

        placas_detectadas = self.detector.plate_detection(imagen)
        if not placas_detectadas:
            self.set_status(404)
            self.write({'error': 'No se encontraron placas en la imagen.'})
            return

        resultado = self.detector.procesar_placas(placas_detectadas)
        #self.write({'placas': resultado})
        print(f'PostPlaca: {resultado}')

def convert_to_serializable(self, data):
        """ Convierte todos los elementos de `data` en tipos serializables por JSON """
        if isinstance(data, np.float32):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()  # Convierte arrays numpy a listas de Python
        elif isinstance(data, dict):
            # Convierte los valores dentro de un diccionario de forma recursiva
            return {key: self.convert_to_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            # Convierte cada elemento de la lista de forma recursiva
            return [self.convert_to_serializable(element) for element in data]
        else:
            return data

def make_app(detector):
    return tornado.web.Application([
        (r"/", DefaultRequestHandler),
        (r"/photo", PhotoRequestHandler, dict(detector=detector)),
    ])

def warm_up_model(detector):
    imagen_dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    detector.plate_detection(imagen_dummy)
    detector.leer(imagen_dummy)

if __name__ == "__main__":
    #model = torch.hub.load('.', 'custom', path='plate_detection.pt', source='local')
    model = YOLO('plate_detection.pt')
    #model.eval()  # Para modo de evaluación
    lectura = easyocr.Reader(['es'], gpu=False)
    detector = PlateDetection(model, lectura)
    warm_up_model(detector)
    app = make_app(detector)
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()

