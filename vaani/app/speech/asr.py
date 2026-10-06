from faster_whisper import WhisperModel
import os

class ASRProcessor:
    def __init__(self, model_size="base"):
        # Run on CPU for simple local setup, change to "cuda" if available
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path):
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return {
            "text": text,
            "language": info.language,
            "probability": info.language_probability
        }
