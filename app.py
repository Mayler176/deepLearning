import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_piezas_ligero.h5", compile=False)

modelo = cargar_modelo()

# Lista de clases (en el orden que usaste)
clases = ['Bishop', 'King', 'Knight', 'Pawn', 'Queen', 'Rook']  # Ajusta si el orden es otro

st.title("♟️ Clasificador de Piezas de Ajedrez")

imagen_subida = st.file_uploader("Sube una imagen de una pieza de ajedrez (85x85 px)", type=["jpg", "png", "jpeg"])

if imagen_subida:
    img = Image.open(imagen_subida).convert('RGB')
    img = img.resize((85, 85))  # Asegura que tenga el tamaño correcto
    st.image(img, caption="Imagen cargada", use_column_width=True)

    # Preprocesamiento
    img_array = np.array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # (1, 85, 85, 3)

    # Predicción
    pred = modelo.predict(img_array)
    clase_predicha = clases[np.argmax(pred)]

    st.markdown(f"### 🔍 Predicción: **{clase_predicha}**")
