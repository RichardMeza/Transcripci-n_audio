import os

# Evitar problemas con torchvision
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import streamlit as st
import whisper
import tempfile
import imageio_ffmpeg

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
# Cargar modelo Whisper
# --------------------------------------------------
@st.cache_resource
def cargar_whisper():

    modelo = whisper.load_model("small")

    return modelo


modelo_whisper = cargar_whisper()

# --------------------------------------------------
# Función resumen extractivo
# --------------------------------------------------
def resumir_texto_largo(texto):

    # Separar oraciones
    oraciones = (
        texto
        .replace("?", ".")
        .replace("¿", "")
        .split(".")
    )

    # Filtrar oraciones muy cortas
    oraciones = [
        o.strip()
        for o in oraciones
        if len(o.strip().split()) > 6
    ]

    if len(oraciones) == 0:

        return (
            "El texto es demasiado corto "
            "para generar un resumen."
        )

    # Palabras importantes
    palabras_clave = [
        "importante",
        "objetivo",
        "concepto",
        "ejemplo",
        "aplicación",
        "herramientas",
        "modelo",
        "datos",
        "análisis",
        "machine learning",
        "python",
        "r",
        "procesamiento",
        "lenguaje natural",
        "predictivo",
        "descriptivo",
        "prescriptivo"
    ]

    puntajes = []

    # Puntuar oraciones
    for oracion in oraciones:

        score = 0

        texto_oracion = oracion.lower()

        for palabra in palabras_clave:

            if palabra in texto_oracion:

                score += 1

        # Bonus por longitud moderada
        score += min(
            len(oracion.split()) / 20,
            2
        )

        puntajes.append(
            (score, oracion)
        )

    # Seleccionar mejores oraciones
    mejores = sorted(
        puntajes,
        key=lambda x: x[0],
        reverse=True
    )[:3]

    resumen = (
        ". ".join(
            [o for _, o in mejores]
        ) + "."
    )

    return resumen

# --------------------------------------------------
# Subida de archivo
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

    # Mostrar audio o video
    if extension == "mp4":

        st.video(archivo)

    else:

        st.audio(archivo)

    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{extension}"
    ) as tmp:

        tmp.write(archivo.read())

        ruta_archivo = tmp.name

    # Botón principal
    if st.button("Transcribir y resumir"):

        # ------------------------------------------
        # Transcripción
        # ------------------------------------------
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

        # ------------------------------------------
        # Resumen
        # ------------------------------------------
        with st.spinner(
            "Generando resumen..."
        ):

            resumen = (
                resumir_texto_largo(texto)
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
