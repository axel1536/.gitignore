import streamlit as Yolo
import requests
from PIL import Image
import io
import cv2
import numpy as np
from collections import Counter

# ────────────────────────────────────────────────
# CONFIGURACIÓN
# ────────────────────────────────────────────────

MODEL_ID = "materials-horz3/1"  # ← Tu modelo real

# Configuración de la página
st.set_page_config(
    page_title="Detector de Materiales de Construcción",
    page_icon="🔧",
    layout="wide"
)

st.title("Detector de Materiales de Construcción")
st.markdown("""
Sube una foto de tu obra y detecta materiales (caños redondos, caños cuadrados, etc.)  
usando tu modelo entrenado en Roboflow: **Materiales 1** (materials-horz3/1)
""")

# Sidebar
with st.sidebar:
    st.header("Opciones")
    conf_threshold = st.slider(
        "Umbral de confianza mínimo",
        0.10, 1.00, 0.40, 0.05,
        help="Más bajo = detecta más objetos (pero puede tener falsos positivos)"
    )

# Subir imagen
uploaded_file = st.file_uploader(
    "Sube una imagen (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Leer imagen
    image = Image.open(uploaded_file)
    st.subheader("Imagen original")
    st.image(image, use_column_width=True)

    if st.button("🔍 Detectar Materiales"):
        with st.spinner("Procesando con Roboflow..."):
            # Preparar imagen para enviar
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            buffered.seek(0)

            # URL de la API de Roboflow
            api_url = (
                f"https://detect.roboflow.com/{MODEL_ID}"
                f"?api_key={st.secrets['ROBOFLOW_API_KEY']}"
                f"&confidence={conf_threshold}"
                f"&overlap=30"  # evita detecciones muy superpuestas
            )

            files = {"file": ("imagen.jpg", buffered, "image/jpeg")}

            try:
                response = requests.post(api_url, files=files, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get("predictions", [])

                    if not predictions:
                        st.warning("No se detectaron materiales en la imagen.")
                    else:
                        # Filtrar por confianza (por si acaso)
                        filtered = [p for p in predictions if p.get("confidence", 0) >= conf_threshold]

                        if not filtered:
                            st.info(f"No hay detecciones con confianza ≥ {conf_threshold:.2f}")
                            filtered = predictions  # mostramos todo si el filtro deja vacío

                        total = len(filtered)
                        conteo_por_clase = Counter(p["class"] for p in filtered)

                        # Resultados principales
                        st.subheader("Resultados")
                        st.metric("**Total de materiales detectados**", total)

                        if conteo_por_clase:
                            cols = st.columns(len(conteo_por_clase))
                            for i, (clase, cantidad) in enumerate(conteo_por_clase.items()):
                                with cols[i]:
                                    st.metric(clase.capitalize(), cantidad)

                        # Detalles
                        with st.expander("Ver todas las detecciones"):
                            for i, p in enumerate(filtered):
                                st.write(
                                    f"• **{p['class']}** — Confianza: **{p['confidence']:.1%}**  "
                                    f"(x: {p['x']:.0f}, y: {p['y']:.0f}, ancho: {p['width']:.0f})"
                                )

                        # Dibujar bounding boxes
                        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        for p in filtered:
                            x1 = int(p["x"] - p["width"] / 2)
                            y1 = int(p["y"] - p["height"] / 2)
                            x2 = int(p["x"] + p["width"] / 2)
                            y2 = int(p["y"] + p["height"] / 2)

                            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{p['class']} {p['confidence']:.2f}"
                            cv2.putText(img_cv, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                        annotated_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        st.subheader("Imagen con detecciones")
                        st.image(annotated_rgb, use_column_width=True)

                else:
                    st.error(f"Error en la API de Roboflow ({response.status_code})")
                    st.code(response.text)

            except requests.exceptions.RequestException as e:
                st.error("No se pudo conectar con Roboflow")
                st.code(str(e))
            except Exception as e:
                st.error("Error inesperado")
                st.code(str(e))

else:
    st.info("Sube una imagen para comenzar.")
