import json
import asyncio
import logging
import websockets
import coloredlogs
from utils.types import ServerState, CACHE_PATH
from utils.songs import initialize_songs
from models.CNN import initialize_model, train_model

# Configure logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s', datefmt='%I:%M:%S %p', isatty=True)

current_state = {
  'state': ServerState.LOADING_SERVER,
  'data': []
}

model = None

def initialize():
  global model
  initialize_songs(current_state)
  model = initialize_model(current_state=current_state)
  train_model(model, current_state=current_state)
  current_state['state'] = ServerState.READY

async def model_info_websocket_handler(websocket):
  global model
  try:
    while True:
      await asyncio.sleep(1)

      # If model is training, append the training progress to the data list
      if current_state['state'] == ServerState.TRAINING_MODEL and type(model) is not dict:
        training_progress = model.get_training_progress()
        current_state['data'].append(training_progress)
      else:
        # If model is not training, clear the data list
        current_state['data'] = []
      
      await websocket.send(json.dumps(current_state))

  except websockets.exceptions.ConnectionClosed:
    await websocket.close()

async def predict_websocket_handler(websocket):
  pass

async def start_model_info_websocket():
  server = await websockets.serve(model_info_websocket_handler, '0.0.0.0', 5000)
  logger.info(f"Model Info Websocket server started on ws://localhost:5000")
  await server.wait_closed()

async def start_predicting_websocket():
  server = await websockets.serve(predict_websocket_handler, '0.0.0.0', 5001)
  logger.info(f"Model Info Websocket server started on ws://localhost:5000")
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