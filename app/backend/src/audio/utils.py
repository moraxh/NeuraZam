import io
from pydub import AudioSegment

def get_audio_from_data(data):
  audio_file = io.BytesIO(data)
  audio = AudioSegment.from_file(audio_file, format="webm")

  mp3_file = io.BytesIO()
  audio.export(mp3_file, format="mp3")  
  mp3_file.seek(0)

  return mp3_file