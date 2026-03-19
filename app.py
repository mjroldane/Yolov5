from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

st.set_page_config(
    page_title="Detección de Objetos Personalizada",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        # Usamos yolov8n o yolov5su según tu preferencia, YOLOv8 es más actual
        model = YOLO("yolov8n.pt") 
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

# --- BARRA LATERAL ---
with st.sidebar:
    # 1. Insertar tu imagen personalizada
    try:
        img_lupa = Image.open("lupaMujer.JPEG")
        st.image(img_lupa, use_container_width=True)
    except:
        st.caption("Asegúrate de que 'lupaMujer.JPEG' esté en la misma carpeta.")

    st.title("Parámetros")
    
    # 2. BUSCADOR ESPECÍFICO
    st.subheader("¿Qué quieres buscar?")
    target_object = st.text_input("Escribe el objeto (en inglés, ej: person, dog, cell phone)", "").lower()
    
    st.divider()
    conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
    iou_threshold  = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
    max_det        = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)

# --- CUERPO PRINCIPAL ---
st.title("🔍 Detección Selectiva de Objetos")
st.markdown("Esta app solo mostrará los objetos que coincidan con tu búsqueda.")

model = load_model()

if model:
    # Obtener el mapeo de nombres del modelo (id: nombre)
    names = model.names
    # Crear un mapeo inverso (nombre: id) para filtrar
    name_to_id = {v.lower(): k for k, v in names.items()}

    picture = st.camera_input("Capturar imagen", key="camera")

    if picture:
        bytes_data = picture.getvalue()
        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img = np.array(pil_img) 

        # Lógica de filtrado por clases
        selected_classes = None
        if target_object in name_to_id:
            selected_classes = [name_to_id[target_object]]
            st.success(f"Buscando únicamente: **{target_object}**")
        elif target_object != "":
            st.warning(f"El objeto '{target_object}' no está en la base de datos. Se mostrarán todos los objetos por defecto.")

        with st.spinner("Analizando..."):
            results = model(
                np_img,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=int(max_det),
                classes=selected_classes # Aquí ocurre la magia del filtro
            )

        result = results[0]
        annotated_rgb = result.plot() # YOLOv8 ya devuelve RGB/BGR según el input

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Resultado de la búsqueda")
            st.image(annotated_rgb, use_container_width=True)

        with col2:
            st.subheader("Resumen de hallazgos")
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                counts = {}
                for box in boxes:
                    label = names[int(box.cls)]
                    counts[label] = counts.get(label, 0) + 1
                
                df = pd.DataFrame([{"Objeto": k, "Cantidad": v} for k, v in counts.items()])
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Objeto")["Cantidad"])
            else:
                st.info("No se encontró lo que buscas en esta imagen.")

else:
    st.error("Error al inicializar el modelo.")

st.markdown("---")
st.caption("Desarrollado con enfoque en Ergonomía Cognitiva para facilitar la búsqueda visual.")
