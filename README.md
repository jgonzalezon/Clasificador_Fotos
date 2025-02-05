# Clasificador de Fotos por Rostros (InsightFace)

Este repositorio contiene un **script único** de Python capaz de **detectar** y **clasificar** rostros en múltiples imágenes. Cada persona se agrupa en su propia carpeta, y si una imagen tiene varios rostros, se copiará en varias carpetas. Además, se guarda un recorte (`representante.jpg`) que sirve como vista previa del rostro en cada carpeta.

Este software está creado con la intención de clasificar las fotos realizadas en un photocall de una fiesta.
Colocar las fotos en input y se clasificarán en la carpeta de clasificados.
---

## Características

- **Detección de rostros** con [InsightFace](https://github.com/deepinsight/insightface).
- **Extracción de embeddings** con ArcFace para alta precisión.
- **Umbral ajustable** (`THRESHOLD`) para controlar la similitud entre rostros.
- **Generación automática de recortes** representativos (uno por persona).
- **Gestión de múltiples rostros** en una sola foto: cada imagen se copia en todas las carpetas de las personas que aparecen.

---

## Requisitos

- Python 3.7 o superior (64 bits recomendado).
- Dependencias clave (puedes instalarlas desde `requirements.txt` si gustas):
  - `numpy`
  - `opencv-python`
  - `insightface`
  - `onnxruntime`

---

## Instalación

1. **Clona** o **descarga** este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/tu-repo.git
   cd tu-repo