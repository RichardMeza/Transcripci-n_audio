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
# Configuración de página
# --------------------------------------------------
st.set_page_config(
    page_title="Transcriptor y Resumidor de Clases",
    page_icon="🎙️",
    layout="centered"
)

st.title("Transcriptor y Resumidor de Clases con IA")

st.write(
    "Sube un audio o video en español. El sistema lo transcribirá con Whisper "
    "y generará un resumen automático usando Transformers."
)

# --------------------------------------------------
# Cargar Whisper
# --------------------------------------------------
@st.cache_resource
def cargar_modelo_whisper():
    return whisper.load_model("small")

modelo_whisper = cargar_modelo_whisper()

# --------------------------------------------------
# Cargar modelo generativo para resumen
# --------------------------------------------------
@st.cache_resource
def cargar_resumidor():
    return pipeline(
        "text-generation",
        model="csebuetnlp/mT5_multilingual_XLSum"
    )

resumidor = cargar_resumidor()

# --------------------------------------------------
# Dividir texto largo
# --------------------------------------------------
def dividir_texto(texto, max_palabras=180):
    palabras = texto.split()
    bloques = []

    for i in range(0, len(palabras), max_palabras):
        bloque = " ".join(palabras[i:i + max_palabras])
        bloques.append(bloque)

    return bloques

# --------------------------------------------------
# Resumir texto
# --------------------------------------------------
def resumir_texto_largo(texto):
    palabras = texto.split()

    if len(palabras) < 40:
        return "El texto es demasiado corto para generar un resumen automático."

    bloques = dividir_texto(texto, max_palabras=180)
    resumenes = []

    for bloque in bloques:
        prompt = (
            "Resume el siguiente texto en español de manera breve y clara:\n\n"
            f"{bloque}\n\nResumen:"
        )

        try:
            salida = resumidor(
                prompt,
                max_new_tokens=120,
                do_sample=False,
                num_return_sequences=1
            )

            resumen = salida[0]["generated_text"]

            # Quitar el prompt si el modelo lo repite
            resumen = resumen.replace(prompt, "").strip()

            if resumen == "":
                resumen = "No se pudo generar resumen para este bloque."

            resumenes.append(resumen)

        except Exception as e:
            resumenes.append(f"Error al resumir bloque: {str(e)}")

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

    extension = archivo.name.split(".")[-1].lower()

    if extension == "mp4":
        st.video(archivo)
    else:
        st.audio(archivo)

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
