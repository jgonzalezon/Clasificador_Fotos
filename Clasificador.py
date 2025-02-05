import os
import shutil
import numpy as np
import cv2

# InsightFace
import insightface
from insightface.app import FaceAnalysis

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
INPUT_DIR = "fotos/input"       # Carpeta con las fotos a clasificar
OUTPUT_DIR = "fotos/clasificadas"
THRESHOLD = 1.1  # Umbral de distancia L2 para considerar dos rostros iguales (ajusta según tus pruebas)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicializa FaceAnalysis (detección y embeddings con ArcFace)
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # o 'CUDAExecutionProvider' si tienes GPU
app.prepare(
    ctx_id=0,           # 0 usa la GPU si está disponible; -1 fuerza CPU
    det_size=(640, 640) # Tamaño para el detector
)

# Estructura de clusters
clusters = []
cluster_count = 1

# Distancia Euclidiana (L2)
def l2_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)


for image_name in os.listdir(INPUT_DIR):
    image_path = os.path.join(INPUT_DIR, image_name)
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue

    print(f"\nProcesando: {image_name}")
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        print("  Error al leer la imagen.")
        continue

    # Detectar rostros y embeddings
    faces = app.get(bgr_img)
    if len(faces) == 0:
        print("  No se detectaron caras.")
        continue

    # Para cada rostro...
    for face in faces:
        # embedding y normalización
        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Comparar con clusters
        matched_cluster = None
        for cluster in clusters:
            distances = [l2_distance(embedding, c_emb) for c_emb in cluster["embeddings"]]
            if np.any(np.array(distances) < THRESHOLD):
                matched_cluster = cluster
                cluster["embeddings"].append(embedding)
                break

        # Si no coincide con nadie, creamos nuevo cluster
        if matched_cluster is None:
            folder_name = f"Persona_{cluster_count:03d}"
            cluster_count += 1
            new_cluster = {
                "folder": folder_name,
                "embeddings": [embedding],
                "images": set(),
                "has_crop": False  # Para saber si ya guardamos un recorte
            }
            clusters.append(new_cluster)
            matched_cluster = new_cluster
            print(f"  => Nuevo cluster creado: {folder_name}")
        else:
            print(f"  => Asignado a {matched_cluster['folder']}")

        # Copiar la imagen original
        if image_name not in matched_cluster["images"]:
            matched_cluster["images"].add(image_name)
            dest_folder = os.path.join(OUTPUT_DIR, matched_cluster["folder"])
            os.makedirs(dest_folder, exist_ok=True)
            shutil.copy(image_path, os.path.join(dest_folder, image_name))
            print(f"     Copiada '{image_name}' en {matched_cluster['folder']}")

        # ──────────────────────────────────────────
        # GUARDAR RECORTE DEL ROSTRO EN EL CLUSTER
        # ──────────────────────────────────────────
        # Queremos dejar "una imagen con un recorte" representativo de la persona.
        # Por ejemplo, guardamos la primera vez que aparece un cluster ("has_crop" = False).
        if not matched_cluster["has_crop"]:
            # Extraemos la caja delimitadora (x1, y1, x2, y2)
            # InsightFace usa .bbox en float (posiciones), redondeamos a int
            x1, y1, x2, y2 = face.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Ajustamos para no salirnos de la imagen
            h, w = bgr_img.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Recortamos
            face_crop = bgr_img[y1:y2, x1:x2]
            crop_path = os.path.join(OUTPUT_DIR, matched_cluster["folder"], "representante.jpg")

            # Guardamos el recorte
            cv2.imwrite(crop_path, face_crop)
            matched_cluster["has_crop"] = True  # Ya guardamos una muestra
            print(f"     Guardado recorte 'representante.jpg' en {matched_cluster['folder']}")

print("\nClasificación completada.")
print(f"Total de clusters: {len(clusters)}")