from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO
import uuid

def convert_text_to_speech(text, path="D:\\", language="en"):
    tts = gTTS(text=text, lang=language, slow=False)
    #tts.save(path)
    path+=str(uuid.uuid4())+".wav"
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)

    # Move the file pointer to the start
    mp3_fp.seek(0)

    # Convert MP3 in memory to WAV in memory
    audio = AudioSegment.from_mp3(mp3_fp)
    wav_fp = BytesIO()
    audio.export(wav_fp, format="wav")

    # Save the WAV file to disk if needed
    with open(path, "wb") as wav_file:
        wav_file.write(wav_fp.getvalue())
    return path
# Text to convert to speech
text = "Hello! This is a text-to-speech conversion example using gTTS."

path = convert_text_to_speech(text)

print("Speech saved to 'output_speech.wav'")