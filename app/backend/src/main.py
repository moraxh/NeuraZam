import os
import logging
import coloredlogs
from fastapi import FastAPI
from utils.songs import initialize_songs
from models.CNN import initialize_model

# Configure logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger, fmt='%(asctime)s [%(levelname)s] %(message)s', datefmt='%I:%M:%S %p', isatty=True)

initialize_songs()
initialize_model()

app = FastAPI(debug=True)