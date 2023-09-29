from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from huggingface_hub import PyTorchModelHubMixin

class ASRInference:  #patrickvonplaten/wav2vec2-base-100h-with-lm  ulrichING/speech_to_text_wave2vect2_english
    def __init__(self, model_name='ulrichING/speech_to_text_wave2vect2_english'):
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)


    def inference(self, audio):
        inputs = self.processor(audio, sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predited_ids = torch.argmax(logits, dim=-1)
        text = self.processor.decode(predited_ids[0]).lower()
        return text