from fastapi import FastAPI
from utilss import ASRInference
from fastapi import UploadFile, File
import soundfile as st


app = FastAPI()

asr = ASRInference()

@app.post("/asr")

def inferebce(file: UploadFile= File(...)):
    audio, _ = st.read(file.file)
    text = asr.inference(audio)

    return {"text": text}