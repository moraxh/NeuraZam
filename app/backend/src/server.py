import json
import base64
import asyncio
import websockets
import pandas as pd
import numpy as np
from models.CNN import get_model
from utils.logger_config import logger
from audio.utils import get_audio_from_data
from utils.constants import SONGS_DATASET_FILE
from audio.features import get_waveform_n_sr_from_file, get_spectogram, get_global_mean_std
from utils.types import current_state, ServerState

async def model_info_websocket_handler(websocket):
  update_interval = 1
  try:
    while True:
      await asyncio.sleep(update_interval)

      # If model is training, append the training progress to the data list
      if current_state['state'] == ServerState.TRAINING_MODEL and get_model() is not None:
        training_progress = get_model().get_training_progress()
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

async def wait_for_model_ready(interval=2):
  while not get_model() or not get_model().is_model_trained:
    await asyncio.sleep(interval)
  
async def predict_websocket_handler(websocket):
  try: 
    logger.info("Waiting for model to be ready...")
    await wait_for_model_ready()
    global_mean, global_std = get_global_mean_std()
    song_info = pd.read_csv(SONGS_DATASET_FILE)
    while True:
      try:
        data = await websocket.recv()
        data = json.loads(data)

        if (data.get("action") == "predict"):
          audio_data = data.get("data")

          if not audio_data:
            await websocket.send(json.dumps({"error": "No audio data provided"}))
            continue

          # Get the audio data from the request
          audio_bytes = base64.b64decode(audio_data)

          try:
            audio_file = get_audio_from_data(audio_bytes)
            waveform, _ = get_waveform_n_sr_from_file(audio_file)
            mel_spec = get_spectogram(waveform)

            # Normalize using the global mean & st
            mel_spec = (mel_spec - global_mean) / global_std

            predictions = get_model().predict(mel_spec)
            pred_id = np.argmax(predictions).item()

            filtered_songs = song_info[song_info['id'] == pred_id]

            if not filtered_songs.empty:
              song = filtered_songs.iloc[0]
              logger.info(f"Predicted Song: {song['name']}")
              await websocket.send(json.dumps({
                "prediction": song.to_dict(),
              }))
          except Exception as e:
            logger.error(f"Error decoding audio data: {e}")
            await websocket.send(json.dumps({"error": "Invalid audio data"}))
      except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON data: {e}")
        await websocket.send(json.dumps({"error": "Invalid JSON data"}))
        continue
  except websockets.exceptions.ConnectionClosed:
    await websocket.close()
  except Exception as e:
    logger.error(f"Error in predict_websocket_handler: {e}")
    await websocket.close()

async def start_model_info_websocket():
  server = await websockets.serve(model_info_websocket_handler, '0.0.0.0', 5000)
  await server.wait_closed()

async def start_predicting_websocket():
  server = await websockets.serve(predict_websocket_handler, '0.0.0.0', 5001)
  await server.wait_closed()