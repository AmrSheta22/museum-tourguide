import whisper
import torch
import os

# Check if a CUDA-enabled GPU is available
def load_tos_model(version):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(version, device=device)
    return model
if __name__ == '__main__':
    audio_file_path = r"D:\repos\museum-tourguide\test\03-01-01-01-01-01-01.wav"
    model = load_tos_model("tiny")
    result = model.transcribe(audio_file_path, fp16=False, language="en")
    print(result["text"])