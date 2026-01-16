import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io
import numpy as np
import cv2
from collections import Counter
MODEL_ID = "materiales-h0rz3/1"

# Configuración de página
st.set_page_config(
    page_title="Detector de Caños - Construcción",
    page_icon="🔧",
    layout="wide"
)

# Cargar cliente de Roboflow (usa secrets para la API Key)
@st.cache_resource
def get_client():
    api_key = st.secrets["ROBOFLOW_API_KEY"]  # Debe estar configurada en Streamlit Cloud → Secrets
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key=api_key
    )
    return client

client = get_client()

# ────────────────────────────────────────────────
# INTERFAZ
# ────────────────────────────────────────────────

st.title("Detector de Caños en Construcción")
st.markdown("""
Sube una foto de tu sitio de obra y detecta/ca cuenta **caños redondos** y **caños cuadrados**  
usando tu modelo entrenado en Roboflow.
""")

# Sidebar con opciones
with st.sidebar:
    st.header("Configuración")
    conf_threshold = st.slider(
        "Umbral de confianza mínimo",
        min_value=0.10,
        max_value=1.00,
        value=0.40,
        step=0.05
    )
    st.caption("Más bajo = más detecciones (pero más falsos positivos)")

# Subir imagen
uploaded_file = st.file_uploader(
    "Sube una imagen (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Mostrar imagen original
    image = Image.open(uploaded_file)
    st.subheader("Imagen subida")
    st.image(image, use_column_width=True)

    # Botón para procesar
    if st.button("🔍 Detectar y Contar Caños"):
        with st.spinner("Enviando imagen a Roboflow para inferencia..."):
            # Preparar imagen en bytes
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()

            try:
                # Inferencia (sin parámetro confidence aquí)
                result = client.infer(
                    img_bytes,
                    model_id=MODEL_ID
                )

                predictions = result.get("predictions", [])

                if not predictions:
                    st.warning("No se detectaron caños en la imagen.")
                else:
                    # Filtrar por umbral de confianza (lo hacemos manualmente)
                    filtered = [p for p in predictions if p.get("confidence", 0) >= conf_threshold]

                    if not filtered:
                        st.warning(f"No se detectaron caños con confianza ≥ {conf_threshold:.2f}")
                        filtered = predictions  # mostramos todos si el filtro deja vacío
                    else:
                        st.success(f"Se detectaron {len(filtered)} caños con confianza ≥ {conf_threshold:.2f}")

                    # Conteo
                    class_names = [p["class"] for p in filtered]
                    conteo_por_clase = Counter(class_names)
                    total = len(filtered)

                    # Resultados principales
                    st.subheader("Resultados")
                    st.metric("**Total de caños detectados**", total)

                    if conteo_por_clase:
                        cols = st.columns(len(conteo_por_clase))
                        for i, (clase, cantidad) in enumerate(conteo_por_clase.items()):
                            with cols[i]:
                                st.metric(clase.capitalize(), cantidad)

                    # Detalles en expander
                    with st.expander("Ver detalles de cada detección"):
                        for i, p in enumerate(filtered):
                            st.write(f"• {p['class']}  —  Confianza: **{p['confidence']:.1%}**  "
                                     f"(x: {p['x']:.0f}, y: {p['y']:.0f}, ancho: {p['width']:.0f})")

                    # Dibujar bounding boxes
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    for p in filtered:
                        x1 = int(p["x"] - p["width"] / 2)
                        y1 = int(p["y"] - p["height"] / 2)
                        x2 = int(p["x"] + p["width"] / 2)
                        y2 = int(p["y"] + p["height"] / 2)

                        color = (0, 255, 0)  # verde
                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                        label = f"{p['class']} {p['confidence']:.2f}"
                        cv2.putText(img_cv, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    annotated_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    st.subheader("Imagen con detecciones")
                    st.image(annotated_rgb, use_column_width=True)

            except Exception as e:
                st.error(f"Error al comunicarse con Roboflow:")
                st.code(str(e))
                st.info("""
                Posibles causas:
                • MODEL_ID incorrecto (revisa formato "proyecto/version")
                • API Key no configurada o inválida (verifica en Secrets)
                • Modelo no desplegado o versión equivocada
                """)

else:
    st.info("Sube una imagen para comenzar.")
