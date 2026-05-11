import streamlit as st
import whisper
import tempfile
import os
import imageio_ffmpeg
from transformers import pipeline

# --------------------------------------------------
# Configuración FFmpeg
# --------------------------------------------------
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

# --------------------------------------------------
# Configuración de la página
# --------------------------------------------------
st.set_page_config(
    page_title="Transcriptor y Resumidor de Clases",
    page_icon="🎙️",
    layout="centered"
)

st.title("Transcriptor y Resumidor de Clases con IA")
st.write(
    "Sube un audio o video en español. El sistema lo transcribirá con Whisper "
    "y luego generará un resumen automático usando Transformers."
)

# --------------------------------------------------
# Cargar modelo Whisper
# --------------------------------------------------
@st.cache_resource
def cargar_modelo_whisper():
    modelo = whisper.load_model("small")
    return modelo

modelo_whisper = cargar_modelo_whisper()

# --------------------------------------------------
# Cargar modelo de resumen
# --------------------------------------------------
@st.cache_resource
def cargar_resumidor():

    resumidor = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    return resumidor

# --------------------------------------------------
# Función para resumir texto largo por bloques
# --------------------------------------------------
def dividir_texto(texto, max_palabras=350):
    palabras = texto.split()
    bloques = []

    for i in range(0, len(palabras), max_palabras):
        bloque = " ".join(palabras[i:i + max_palabras])
        bloques.append(bloque)

    return bloques


def resumir_texto_largo(texto):
    bloques = dividir_texto(texto, max_palabras=350)
    resumenes = []

    for bloque in bloques:
        if len(bloque.split()) >= 40:
            resumen = resumidor(
                bloque,
                max_length=120,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]

            resumenes.append(resumen)

    resumen_final = " ".join(resumenes)

    return resumen_final

# --------------------------------------------------
# Subida de archivo
# --------------------------------------------------
archivo = st.file_uploader(
    "Sube tu audio o video",
    type=["mp3", "wav", "m4a", "ogg", "flac", "mp4"]
)

if archivo is not None:

    st.subheader("Archivo cargado")
    st.audio(archivo)

    extension = archivo.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
        tmp.write(archivo.read())
        ruta_archivo = tmp.name

    if st.button("Transcribir y resumir"):

        with st.spinner("Transcribiendo con Whisper..."):
            resultado = modelo_whisper.transcribe(
                ruta_archivo,
                language="es"
            )

            texto = resultado["text"]

        st.subheader("Transcripción")
        st.write(texto)

        st.download_button(
            label="Descargar transcripción",
            data=texto,
            file_name="transcripcion.txt",
            mime="text/plain"
        )

        with st.spinner("Generando resumen automático..."):
            resumen = resumir_texto_largo(texto)

        st.subheader("Resumen automático")
        st.write(resumen)

        st.download_button(
            label="Descargar resumen",
            data=resumen,
            file_name="resumen.txt",
            mime="text/plain"
        )

    if os.path.exists(ruta_archivo):
        os.remove(ruta_archivo)
