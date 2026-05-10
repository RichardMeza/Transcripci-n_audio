import streamlit as st
import whisper
import tempfile
import os
import imageio_ffmpeg

os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

# --------------------------------------------------
# Configuración de la página
# --------------------------------------------------
st.set_page_config(
    page_title="Transcriptor de Audio en Español",
    page_icon="🎙️",
    layout="centered"
)

st.title("Transcriptor de Audio en Español con Whisper")
st.write("Sube un audio en formato MP3, WAV, M4A, OGG o FLAC y el sistema lo transcribirá automáticamente.")

# --------------------------------------------------
# Cargar modelo Whisper
# --------------------------------------------------
@st.cache_resource
def cargar_modelo():
    modelo = whisper.load_model("small")
    return modelo

modelo = cargar_modelo()

# --------------------------------------------------
# Subida de audio
# --------------------------------------------------
audio_file = st.file_uploader(
    "Sube tu audio",
    type=["mp3", "wav", "m4a", "ogg", "flac","mp4"]
)

if audio_file is not None:

    st.subheader("Audio cargado")
    st.audio(audio_file)

    # Guardar temporalmente el audio
    extension = audio_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
        tmp.write(audio_file.read())
        ruta_audio = tmp.name

    # Botón para transcribir
    if st.button("Transcribir audio"):

        with st.spinner("Transcribiendo audio..."):
            resultado = modelo.transcribe(
                ruta_audio,
                language="es"
            )

            texto = resultado["text"]

        st.subheader("Transcripción")
        st.write(texto)

        # Descargar texto
        st.download_button(
            label="Descargar transcripción",
            data=texto,
            file_name="transcripcion.txt",
            mime="text/plain"
        )

    # Eliminar archivo temporal
    if os.path.exists(ruta_audio):
        os.remove(ruta_audio)