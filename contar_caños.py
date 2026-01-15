from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

# === CONFIGURACIÓN ===
MODEL_PATH = 'runs/detect/train/weights/best.pt'  # Cambia esto a la ruta REAL de tu best.pt
# Ejemplos comunes:
# - Si entrenaste en Roboflow/Colab: 'runs/detect/train/weights/best.pt'
# - O descarga de Roboflow: 'path/to/best.pt'

IMAGEN_PRUEBA = 'ruta/a/tu/imagen_de_prueba.jpg'  # Pon aquí una foto con varios caños
# O usa: 'test.jpg' si la tienes en la misma carpeta

CONF_THRESHOLD = 0.4  # Umbral de confianza (ajústalo si detecta mucho ruido)

# Cargar el modelo
print("Cargando modelo YOLOv8...")
model = YOLO(MODEL_PATH)
print("Modelo cargado exitosamente.")
print("Clases del modelo:", model.names)  # Debería mostrar {0: 'Caños redondos', 1: 'Caños cuadrados'} o similar

# Cargar la imagen
img = cv2.imread(IMAGEN_PRUEBA)
if img is None:
    print(f"Error: No se pudo cargar la imagen {IMAGEN_PRUEBA}")
    exit()

print(f"Procesando imagen: {IMAGEN_PRUEBA}")
print(f"Tamaño de la imagen: {img.shape}")

# Realizar la predicción
results = model.predict(img, conf=CONF_THRESHOLD, verbose=True)

# Extraer las detecciones de la primera (y única) imagen
detections = results[0]

# Si no hay detecciones
if len(detections.boxes) == 0:
    print("No se detectaron caños en la imagen.")
else:
    # Obtener IDs de clase (0 o 1, según tu modelo)
    class_ids = detections.boxes.cls.cpu().numpy().astype(int)
    
    # Mapear a nombres de clases
    class_names = [model.names[cid] for cid in class_ids]
    
    # Conteo por clase
    conteo_por_clase = Counter(class_names)
    
    # Conteo total
    total_caños = len(class_ids)
    
    # Mostrar resultados
    print("\n=== RESULTADOS DE DETECCIÓN ===")
    print(f"Total de caños detectados: {total_caños}")
    print("Desglose por tipo:")
    for clase, cantidad in conteo_por_clase.items():
        print(f"  - {clase}: {cantidad}")
    
    # Opcional: mostrar confianza promedio por clase
    print("\nConfianzas detalladas (opcional):")
    for i, box in enumerate(detections.boxes):
        clase = model.names[int(box.cls)]
        conf = box.conf.item()
        print(f"  Caño {i+1}: {clase} - Confianza: {conf:.2f}")

    # Guardar imagen con anotaciones (opcional)
    annotated_img = detections.plot()  # Dibuja bounding boxes
    cv2.imwrite('resultado_anotado.jpg', annotated_img)
    print("\nImagen anotada guardada como 'resultado_anotado.jpg'")
