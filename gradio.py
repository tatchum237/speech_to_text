import gradio as gr
from utilss import ASRInference
import speech_recognition as sr


asr = ASRInference()

def speech_to_text(audio_clip):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_clip) as source:
        audio = recognizer.record(source)
    try:
        text = asr.inference(audio)
        return text
    except sr.UnknownValueError:
        return "Impossible de reconnaître la parole."
    except sr.RequestError as e:
        return f"Erreur lors de la demande de reconnaissance : {e}"


audio_input = gr.inputs.Audio(label="Chargez un fichier audio")
text_output = gr.outputs.Textbox(label="Texte transcrit")

gr.Interface(
    fn=speech_to_text,
    inputs=audio_input,
    outputs=text_output,
    live=True,
    title="Speech-to-Text",
    description="Transcription automatique de la parole en texte.",
).launch()
