import asyncio
from models.CNN import initialize_model
from audio.songs import initialize_songs
from audio.features import initialize_extract_features
from server import start_model_info_websocket, start_predicting_websocket

def initialize():
  initialize_songs()
  initialize_extract_features()
  initialize_model()

async def run_main():
  asyncio.create_task(start_model_info_websocket())
  asyncio.create_task(start_predicting_websocket())

  await asyncio.to_thread(initialize)

  await asyncio.Future()  # Run forever

def main():
  asyncio.run(run_main())

if __name__ == "__main__":
  main()