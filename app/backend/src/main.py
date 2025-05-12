import json
import asyncio
import logging
import websockets
import coloredlogs
from utils.types import ServerState, ValidationException
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
  update_interval = 1
  global model
  try:
    while True:
      await asyncio.sleep(update_interval)

      # If model is training, append the training progress to the data list
      if current_state['state'] == ServerState.TRAINING_MODEL and type(model) is not dict:
        training_progress = model.get_training_progress()
        current_state['data'].append(training_progress)
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
      logging.info(f"Received data: {data}")
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