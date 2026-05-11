import os

# Evitar problemas con torchvision
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import streamlit as st
import whisper
import tempfile
import imageio_ffmpeg

from transformers import pipeline

# --------------------------------------------------
# Configuración FFmpeg
# --------------------------------------------------
os.environ["PATH"] += (
    os.pathsep +
    os.path.dirname(
        imageio_ffmpeg.get_ffmpeg_exe()
    )
)

# --------------------------------------------------
# Configuración página
# --------------------------------------------------
st.set_page_config(
    page_title="Transcriptor y Resumidor IA",
    page_icon="🎙️",
    layout="centered"
)

st.title("Transcriptor y Resumidor de Clases con IA")

st.write(
    "Sube un audio o video en español. "
    "El sistema transcribirá el contenido "
    "y luego generará un resumen automático."
)

# --------------------------------------------------
# Cargar Whisper
# --------------------------------------------------
@st.cache_resource
def cargar_whisper():

    modelo = whisper.load_model("small")

    return modelo


modelo_whisper = cargar_whisper()

# --------------------------------------------------
# Cargar modelo resumen
# --------------------------------------------------
@st.cache_resource
def cargar_resumidor():

    resumidor = pipeline(
        "text-generation",
        model="google/flan-t5-base"
    )

    return resumidor


resumidor = cargar_resumidor()

# --------------------------------------------------
# Dividir texto largo
# --------------------------------------------------
def dividir_texto(texto, max_palabras=180):

    palabras = texto.split()

    bloques = []

    for i in range(
        0,
        len(palabras),
        max_palabras
    ):

        bloque = " ".join(
            palabras[i:i + max_palabras]
        )

        bloques.append(bloque)

    return bloques

# --------------------------------------------------
# Resumir texto largo
# --------------------------------------------------
def resumir_texto_largo(texto):

    palabras = texto.split()

    if len(palabras) < 40:

        return (
            "El texto es demasiado corto "
            "para generar un resumen."
        )

    bloques = dividir_texto(
        texto,
        max_palabras=180
    )

    resumenes = []

    for bloque in bloques:

        prompt = f"""
Genera un resumen corto y claro en español del siguiente texto.

Texto:
{bloque}

Resumen:
"""

        try:

            salida = resumidor(
                prompt,
                max_new_tokens=120,
                do_sample=False
            )

            resumen = (
                salida[0]["generated_text"]
                .strip()
            )

            if "Resumen:" in resumen:

                resumen = (
                    resumen
                    .split("Resumen:")[-1]
                    .strip()
                )

            if resumen == "":

                resumen = (
                    "No se pudo generar "
                    "resumen."
                )

            resumenes.append(resumen)

        except Exception as e:

            resumenes.append(
                f"Error: {str(e)}"
            )

    resumen_final = " ".join(resumenes)

    return resumen_final

# --------------------------------------------------
# Subir archivo
# --------------------------------------------------
archivo = st.file_uploader(
    "Sube tu audio o video",
    type=[
        "mp3",
        "wav",
        "m4a",
        "ogg",
        "flac",
        "mp4"
    ]
)

# --------------------------------------------------
# Procesamiento
# --------------------------------------------------
if archivo is not None:

    st.subheader("Archivo cargado")

    extension = (
        archivo.name
        .split(".")[-1]
        .lower()
    )

    if extension == "mp4":

        st.video(archivo)

    else:

        st.audio(archivo)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{extension}"
    ) as tmp:

        tmp.write(archivo.read())

        ruta_archivo = tmp.name

    if st.button("Transcribir y resumir"):

        # ----------------------------
        # Transcripción
        # ----------------------------
        with st.spinner(
            "Transcribiendo con Whisper..."
        ):

            resultado = (
                modelo_whisper.transcribe(
                    ruta_archivo,
                    language="es"
                )
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

        # ----------------------------
        # Resumen
        # ----------------------------
        with st.spinner(
            "Generando resumen..."
        ):

            resumen = resumir_texto_largo(
                texto
            )

        st.subheader(
            "Resumen automático"
        )

        st.write(resumen)

        st.download_button(
            label="Descargar resumen",
            data=resumen,
            file_name="resumen.txt",
            mime="text/plain"
        )

    # --------------------------------------------------
    # Eliminar archivo temporal
    # --------------------------------------------------
    if os.path.exists(ruta_archivo):

        os.remove(ruta_archivo)
