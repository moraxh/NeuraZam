import logging
from fastapi import FastAPI

from utils.songs import process_songs

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check if the songs are already downloaded if not download them
process_songs()

app = FastAPI(debug=True)