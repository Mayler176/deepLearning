import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import joblib

# Menú de navegación
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Selecciona una sección", 
                          ["📘 Acerca de", "🧠 Clasificación de Imágenes", "💬 Análisis de Sentimiento", "📈 Regresión"])

# ------------ PÁGINA 1: ACERCA DE ------------ #
if opcion == "📘 Acerca de":
    st.title("🧠 Proyecto de Deep Learning")
    st.markdown("""
    Este proyecto incluye tres modelos de aprendizaje profundo:

    - Clasificación de imágenes de piezas de ajedrez
    - Análisis de sentimiento en texto
    - Predicción numérica por regresión

    Desarrollado por Ana ✨
    """)

# ------------ PÁGINA 2: CLASIFICACIÓN DE IMÁGENES ------------ #
elif opcion == "🧠 Clasificación de Imágenes":
    st.title("♟️ Clasificador de Piezas de Ajedrez")

    @st.cache_resource
    def cargar_modelo_imagen():
        return tf.keras.models.load_model("modelo_piezas_ligero.h5")

    modelo_img = cargar_modelo_imagen()
    clases = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']

    archivo = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
    if archivo:
        img = Image.open(archivo).convert("RGB")
        st.image(img, caption="Imagen subida", use_column_width=True)
        img_array = np.array(img.resize((85, 85))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = modelo_img.predict(img_array)
        st.success(f"Pieza predicha: **{clases[np.argmax(pred)]}**")

# ------------ PÁGINA 3: ANÁLISIS DE SENTIMIENTO ------------ #
elif opcion == "💬 Análisis de Sentimiento":
    st.title("💬 Análisis de Sentimiento")
    
    @st.cache_resource
    def cargar_modelo_sentimiento():
        return tf.keras.models.load_model("analisis_sentimiento.h5")

    modelo_sent = cargar_modelo_sentimiento()
    
    texto = st.text_area("Escribe un mensaje para analizar el sentimiento")

    if texto:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import Tokenizer

        # Tokenización ejemplo simple
        tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        tokenizer.fit_on_texts([texto])
        secuencia = tokenizer.texts_to_sequences([texto])
        entrada = pad_sequences(secuencia, maxlen=50)

        pred = modelo_sent.predict(entrada)
        st.write("Resultado:")
        st.success("Positivo" if np.argmax(pred) == 1 else "Negativo")

# ------------ PÁGINA 4: REGRESIÓN ------------ #
elif opcion == "📈 Regresión":
    st.title("📈 Predicción por Regresión")

    @st.cache_resource
    def cargar_modelo_regresion():
        return tf.keras.models.load_model("regression.h5")


    modelo_reg = cargar_modelo_regresion()

    st.markdown("Introduce los valores para predecir el resultado (ej. precio):")
    col1, col2 = st.columns(2)
    feature1 = col1.number_input("Característica 1", value=0.0)
    feature2 = col2.number_input("Característica 2", value=0.0)
    # agrega más si necesitas

    if st.button("Predecir"):
        entrada = np.array([[feature1, feature2]])  # ajusta dimensiones
        pred = modelo_reg.predict(entrada)
        st.success(f"Valor predicho: **{pred[0]:.2f}**")
