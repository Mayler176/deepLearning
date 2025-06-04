import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image

# Título
st.title("♟️ Clasificador de Piezas de Ajedrez")

# Instrucciones
st.markdown("Sube una imagen de una pieza de ajedrez para identificar de cuál se trata.")

# Selector de modelo
tipo_modelo = st.selectbox("¿Qué tipo de modelo quieres usar?", ["Modelo .h5 (Keras)", "Modelo .pkl (Pickle)"])

# Cargar modelo
@st.cache_resource
def cargar_modelo_h5(path="modelo_piezas_ajedrez.h5"):
    return tf.keras.models.load_model(path)

@st.cache_resource
def cargar_modelo_pkl(path="modelo_piezas_ajedrez.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

if tipo_modelo == "Modelo .h5 (Keras)":
    modelo = cargar_modelo_h5()
else:
    modelo = cargar_modelo_pkl()

# Etiquetas (ajusta esto según tu modelo)
etiquetas = ['alfil', 'caballo', 'peon', 'reina', 'rey', 'torre']

# Subir imagen
imagen = st.file_uploader("Sube la imagen", type=["jpg", "png", "jpeg"])

if imagen:
    img = Image.open(imagen).convert("RGB")
    st.image(img, caption="Imagen subida", use_column_width=True)

    # Preprocesamiento
    img_resized = img.resize((224, 224))  # Cambia a lo que tu modelo necesite
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    if tipo_modelo == "Modelo .h5 (Keras)":
        pred = modelo.predict(img_array)
    else:  # Para modelos entrenados con scikit-learn, que esperan vectores
        img_flat = img_array.flatten().reshape(1, -1)
        pred = modelo.predict_proba(img_flat)

    clase_predicha = etiquetas[np.argmax(pred)]
    st.success(f"La pieza detectada es: **{clase_predicha.upper()}**")
