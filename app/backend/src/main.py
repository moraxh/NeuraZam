import io
import json
import faiss
import torch
import asyncio
import logging
import websockets
import torchaudio
import coloredlogs
from pydub import AudioSegment
from utils.songs import initialize_songs
from models.CNN import EMBEDDINGS_FILE, EMBEDDINGS_METADATA_FILE, initialize_model, train_model
from utils.types import ServerState, ValidationException
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# Configure logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s', datefmt='%I:%M:%S %p', isatty=True)

current_state = {
  'state': ServerState.LOADING_SERVER,
  'data': []
}

model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize():
  global model
  initialize_songs(current_state)
  model = initialize_model(current_state=current_state)
  train_model(model, current_state=current_state)
  current_state['state'] = ServerState.READY

async def model_info_websocket_handler(websocket):
  update_interval = 1
  global model
  try:
    while True:
      await asyncio.sleep(update_interval)

      # If model is training, append the training progress to the data list
      if current_state['state'] == ServerState.TRAINING_MODEL and type(model) is not dict:
        training_progress = model.get_training_progress()
        current_state['data'] = training_progress
      elif current_state['state'] == ServerState.READY:
        # If model is ready, send the current state at less frequent intervals
        update_interval = 5
      else:
        # If model is not training, clear the data list
        current_state['data'] = []
      
      await websocket.send(json.dumps(current_state))
  except websockets.exceptions.ConnectionClosed:
    await websocket.close()
  except Exception as e:
    logger.error(f"Error in model_info_websocket_handler: {e}")
    await websocket.close()

async def predict_websocket_handler(websocket):
  try:
    while True:
      data = await websocket.recv()

      if (model.is_model_trained == False):
        await websocket.send(json.dumps({"error": "Model is not trained yet"}))
        continue

      if (isinstance(data, bytes)):

        try:
          audio_file = io.BytesIO(data)
          audio = AudioSegment.from_file(audio_file, format="webm")

          wav_file = io.BytesIO()
          audio.export(wav_file, format="mp3")
          wav_file.seek(0)

          waveform, sample_rate = torchaudio.load(wav_file)
          waveform = waveform.mean(dim=0, keepdim=True)  # Mono
          waveform = waveform.unsqueeze(0)  # [B, C, T] => [1, 1, T]
          waveform = waveform / waveform.abs().max() # Normalize
          waveform = waveform.to(device)  

          mel_spec_transform = torch.nn.Sequential(
            MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=64),
            AmplitudeToDB()
          )
          mel_spec_transform = torch.jit.script(mel_spec_transform.to(device))  # Compile the model

          with torch.no_grad():
            mel_spec = mel_spec_transform(waveform)
            mel_spec = mel_spec.squeeze(0).cpu().numpy()

          result = model.predict(mel_spec)
          result = torch.nn.functional.normalize(result, p=2, dim=1) # L2 normalization

          # Get others embeddings
          index = faiss.read_index(EMBEDDINGS_FILE)

          with open(EMBEDDINGS_METADATA_FILE, 'r') as f:
            metadata = json.load(f)

          similarity, indexes = index.search(result, 5)

          indexes = indexes[0]

          print(indexes)


        except Exception as e:
          logging.error(f"Invalid audio data: {e}")
          await websocket.send(json.dumps({"error": "Invalid audio data"}))
          continue
  except websockets.exceptions.ConnectionClosed:
    await websocket.close()
  except ValidationException as e:
    logger.error(f"Validation error: {e}")
    await websocket.send(json.dumps({"error": "Validation error"}))
  except Exception as e:
    logger.error(f"Error in predict_websocket_handler: {e}")
    await websocket.close()

async def start_model_info_websocket():
  server = await websockets.serve(model_info_websocket_handler, '0.0.0.0', 5000)
  await server.wait_closed()

async def start_predicting_websocket():
  server = await websockets.serve(predict_websocket_handler, '0.0.0.0', 5001)
  await server.wait_closed()

async def run_main():
  asyncio.create_task(start_model_info_websocket())
  asyncio.create_task(start_predicting_websocket())

  await asyncio.to_thread(initialize)

  await asyncio.Future()

def main():
  asyncio.run(run_main())

if __name__ == "__main__":
  main()