import os
import base64
import requests

def imagen_a_base64(ruta_imagen):
    with open(ruta_imagen, 'rb') as img_file:
        imagen_bytes = img_file.read()
        imagen_base64 = base64.b64encode(imagen_bytes).decode('utf-8')
    return imagen_base64

def enviar_imagen_base64(url, imagen_base64, tipo_imagen):
    payload = {
        'foto': f'data:image/{tipo_imagen};base64,{imagen_base64}',
        'tipo': tipo_imagen
    }
    response = requests.post(url, json=payload)
    return response.json()

def detectar_placas_en_carpeta(carpeta, url):
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta_imagen = os.path.join(carpeta, archivo)
            tipo_imagen = archivo.split('.')[-1]  # Obtiene la extensión del archivo
            imagen_base64 = imagen_a_base64(ruta_imagen)
            try:
                resultado = enviar_imagen_base64(url, imagen_base64, tipo_imagen)
                print(f'Resultado para {archivo}: {resultado}')
            except Exception as e:
                print(f'Error al procesar {archivo}: {e}')

if __name__ == "__main__":
    carpeta_fotos = 'fotos'  # Ruta de la carpeta donde están las fotos
    url_servidor = 'http://localhost:8888/photo'  # URL del servidor Tornado
    detectar_placas_en_carpeta(carpeta_fotos, url_servidor)
