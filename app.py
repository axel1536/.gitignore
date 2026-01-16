import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io
import numpy as np
import cv2
from collections import Counter

# Configuración de página
st.set_page_config(
    page_title="Detector de Caños - Roboflow",
    page_icon="🔧",
    layout="wide"
)

# Cargar cliente de Roboflow (usa secrets para la key)
@st.cache_resource
def get_client():
    api_key = st.secrets["ROBOFLOW_API_KEY"]  # Agrega esto en Streamlit Cloud > Secrets
    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",  # o "https://serverless.roboflow.com" si usas serverless
        api_key=api_key
    )
    return client

client = get_client()

MODEL_ID = "tu-modelo-id/version"  # CAMBIA ESTO: ej "axel-caños-construccion/1"

st.title("Detector de Caños en Construcción con Roboflow")
st.markdown("Sube una foto y detecta/ca cuenta caños redondos y cuadrados usando tu modelo entrenado en Roboflow.")

# Sidebar
with st.sidebar:
    st.header("Opciones")
    conf_threshold = st.slider("Umbral de confianza", 0.1, 1.0, 0.4, 0.05)

uploaded_file = st.file_uploader("Sube una imagen (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("Imagen original")
    st.image(image, use_column_width=True)

    if st.button("Detectar y Contar Caños"):
        with st.spinner("Enviando a Roboflow para inferencia..."):
            # Preparar imagen como bytes
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()

            # Inferencia
            try:
                predictions = client.infer(
                    img_bytes,
                    model_id=MODEL_ID,
                    confidence=conf_threshold,
                    # Opcional: si quieres imagen anotada de vuelta
                    # visualize=True  # Algunos SDK lo soportan, prueba
                )

                preds = predictions["predictions"]  # Lista de detecciones

                if not preds:
                    st.warning("No se detectaron caños. Prueba bajando el umbral.")
                else:
                    # Conteo
                    class_names = [p["class"] for p in preds]
                    conteo_por_clase = Counter(class_names)
                    total = len(preds)

                    # Mostrar métricas bonitas
                    st.subheader("Resultados")
                    st.metric("Total de caños detectados", total)

                    cols = st.columns(len(conteo_por_clase) or 1)
                    for idx, (clase, cant) in enumerate(conteo_por_clase.items()):
                        with cols[idx % len(cols)]:
                            st.metric(clase.capitalize(), cant)

                    # Detalles
                    with st.expander("Detalles de detecciones"):
                        for i, p in enumerate(preds):
                            st.write(f"- {p['class']} → Confianza: {p['confidence']:.2%} | Posición: x={p['x']:.0f}, y={p['y']:.0f}")

                    # Mostrar imagen anotada (si el SDK devuelve base64 o usa cv2 para dibujar)
                    # Alternativa simple: dibujar manualmente
                    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    for p in preds:
                        x, y, w, h = int(p["x"] - p["width"]/2), int(p["y"] - p["height"]/2), int(p["width"]), int(p["height"])
                        cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(img_cv, f"{p['class']} {p['confidence']:.2f}", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    annotated_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    st.subheader("Imagen con detecciones")
                    st.image(annotated_rgb, use_column_width=True)

            except Exception as e:
                st.error(f"Error en inferencia: {str(e)} - Verifica MODEL_ID y API_KEY.")
