import streamlit as st
import numpy as np
import pandas as pd
import torch
import io
from PIL import Image
from ultralytics import YOLO

# --- Configuración Inicial ---
st.set_page_config(page_title="Buscador Inteligente", layout="wide")

@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt") # YOLOv8 es más rápido y preciso

model = load_yolo()

# --- BARRA LATERAL (Donde va la imagen y el buscador) ---
with st.sidebar:
    # Intentamos cargar la imagen con las dos extensiones más comunes
    try:
        # Probamos con .jpg (como se ve en tu captura)
        img_lupa = Image.open("lupaMujer.jpg")
        st.image(img_lupa, use_container_width=True)
    except FileNotFoundError:
        try:
            # Si falla, probamos con .JPEG (por si acaso)
            img_lupa = Image.open("lupaMujer.JPEG")
            st.image(img_lupa, use_container_width=True)
        except FileNotFoundError:
            st.error("⚠️ No se encontró 'lupaMujer.jpg'. Verifica que esté en la misma carpeta que este script.")

    st.title("Filtro de Búsqueda")
    
    # Campo para escribir qué queremos buscar
    target = st.text_input("¿Qué objeto buscas?", placeholder="Ej: person, cell phone, cup").lower().strip()
    
    st.divider()
    conf_level = st.slider("Sensibilidad (Confianza)", 0.0, 1.0, 0.25)

# --- CUERPO PRINCIPAL ---
st.title("🔍 Detector Selectivo")

picture = st.camera_input("Toma una foto")

if picture and model:
    # Procesar imagen
    img = Image.open(io.BytesIO(picture.getvalue())).convert("RGB")
    img_array = np.array(img)

    # Lógica de filtrado: Obtener IDs de clases
    # YOLO maneja nombres como 'person', 'cell phone', etc.
    class_ids = None
    if target:
        # Buscamos el ID numérico que corresponde al texto escrito
        class_ids = [id for id, name in model.names.items() if name.lower() == target]
        
        if not class_ids:
            st.warning(f"El objeto '{target}' no existe en la base de datos de YOLO. Mostrando todo por defecto.")
            class_ids = None
        else:
            st.success(f"Filtrando resultados para: **{target}**")

    # Predicción
    results = model(img_array, conf=conf_level, classes=class_ids)
    
    # Dibujar resultados
    annotated_img = results[0].plot()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(annotated_img, caption="Resultado de la búsqueda", use_container_width=True)
    
    with col2:
        st.subheader("Estadísticas")
        if len(results[0].boxes) > 0:
            # Contar detecciones
            found_names = [model.names[int(b.cls)] for b in results[0].boxes]
            df = pd.DataFrame(found_names, columns=["Objeto"]).value_counts().reset_index()
            df.columns = ["Objeto", "Cantidad"]
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No se encontró el objeto buscado.")
