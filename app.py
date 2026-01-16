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

        # Inferencia SIN 'confidence' en infer()
        try:
            predictions = client.infer(
                img_bytes,
                model_id=MODEL_ID
                # QUITA ESTO: , confidence=conf_threshold
            )

            preds = predictions.get("predictions", [])  # Asegura que sea lista

            if not preds:
                st.warning("No se detectaron caños. Prueba con otra foto o verifica si el modelo está activo en Roboflow.")
            else:
                # Filtrar manualmente por confianza si quieres (ej: > 0.4)
                filtered_preds = [p for p in preds if p.get("confidence", 0) >= conf_threshold]

                if not filtered_preds:
                    st.warning(f"No se detectaron caños con confianza >= {conf_threshold}. Baja el umbral o usa otra imagen.")
                    filtered_preds = preds  # Usa todos si no hay después del filtro

                # Conteo con los filtrados
                class_names = [p["class"] for p in filtered_preds]
                conteo_por_clase = Counter(class_names)
                total = len(filtered_preds)

                # Mostrar resultados
                st.subheader("Resultados")
                st.metric("Total de caños detectados", total)

                if conteo_por_clase:
                    cols = st.columns(len(conteo_por_clase))
                    for idx, (clase, cant) in enumerate(conteo_por_clase.items()):
                        with cols[idx]:
                            st.metric(clase.capitalize(), cant)
                else:
                    st.info("Ninguna clase detectada después del filtro.")

                # Detalles
                with st.expander("Detalles de detecciones"):
                    for i, p in enumerate(filtered_preds):
                        st.write(f"- {p['class']} → Confianza: {p['confidence']:.2%} | Posición: x={p['x']:.0f}, y={p['y']:.0f}")

                # Dibujar bounding boxes manualmente
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                for p in filtered_preds:
                    x1 = int(p["x"] - p["width"] / 2)
                    y1 = int(p["y"] - p["height"] / 2)
                    x2 = int(p["x"] + p["width"] / 2)
                    y2 = int(p["y"] + p["height"] / 2)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{p['class']} {p['confidence']:.2f}"
                    cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                annotated_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                st.subheader("Imagen con detecciones")
                st.image(annotated_rgb, use_column_width=True)

        except Exception as e:
            st.error(f"Error en inferencia: {str(e)}")
            st.info("Verifica: 1) MODEL_ID correcto (ej: 'tu-proyecto/1'), 2) API_KEY en Secrets, 3) Modelo activo en Roboflow Deploy.")
Pasos para aplicar el fix
