from gtts import gTTS
import os
import uuid

class TTSProcessor:
    def __init__(self, output_dir="vaani/temp_audio"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def synthesize(self, text, lang='en'):
        """
        lang: 'en' for English, 'kn' for Kannada
        """
        try:
            tts = gTTS(text=text, lang=lang)
            filename = f"response_{uuid.uuid4()}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            tts.save(filepath)
            return filepath
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
