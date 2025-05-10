import logging
from fastapi import FastAPI

from utils.songs import download_songs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if the songs are already downloaded if not download them
download_songs()

app = FastAPI(debug=True)