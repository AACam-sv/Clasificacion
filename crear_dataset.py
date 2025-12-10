import cv2
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- CONFIGURACIÓN ---
ACTIONS = np.array(['hola', 'gracias', 'espera', 'ayuda', 'buenos_dias', 
                    'donde', 'quien', 'que', 'cuando', 'por_favor'])
NO_SEQUENCES = 30  # Cuantas veces repetirás cada gesto (videos)
SEQUENCE_LENGTH = 20 # Cuantos frames dura cada video
# ---------------------

# Cargar extractor de características (CNN)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Crear carpeta de datos
DATA_PATH = os.path.join('MP_Data') 
for action in ACTIONS: 
    for sequence in range(NO_SEQUENCES):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except: pass

cap = cv2.VideoCapture(0)

def extract_features(frame):
    # Asumimos que la mano está en el centro o usas todo el frame para ir rápido
    # Lo ideal es recortar la mano aquí si tienes YOLO/MediaPipe funcionando
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return base_model.predict(img, verbose=0).flatten()

print("¡PREPÁRATE! Toca 'ESPACIO' para empezar a grabar cada clase.")

for action_num, action in enumerate(ACTIONS):
    print(f'Recolectando datos para: {action}')
    # Esperar confirmación del usuario antes de empezar una nueva clase
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Presiona ESPACIO para grabar "{action}"', (20,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow('Recorder', frame)
        if cv2.waitKey(10) == 32: # 32 es Espacio
            break
            
    for sequence in range(NO_SEQUENCES):
        # Un pequeño respiro entre videos
        for i in range(10): # 10 frames de pausa visual
            ret, frame = cap.read()
            cv2.putText(frame, f'GRABANDO VIDEO {sequence+1}/{NO_SEQUENCES}', (20,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Recorder', frame)
            cv2.waitKey(10)

        # Grabación real de la secuencia
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            
            # 1. Extraer características (AQUÍ OCURRE LA MAGIA DE LA CNN)
            features = extract_features(frame)
            
            # 2. Guardar el vector npy
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, features)

            # Visualizar
            cv2.putText(frame, f'Grabando... {frame_num}', (20,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Recorder', frame)
            cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()