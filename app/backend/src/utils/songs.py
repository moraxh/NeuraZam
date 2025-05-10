import os
import subprocess
import logging

# Must be a valid Spotify public playlist url
SPOTIFY_PLAYLIST_URL = os.getenv("SPOTIFY_PLAYLIST_URL")
SONGS_DIR = os.getenv("SONGS_DIR", "songs")

if not SPOTIFY_PLAYLIST_URL:
    raise ValueError("SPOTIFY_PLAYLIST_URL must be set in the environment variables")

def download_songs():
    """
    Download songs from the Spotify playlist URL using spotdl(https://github.com/spotDL/spotify-downloader)
    """

    logging.info("Checking if songs are already downloaded...")

    # Check if spotdl is installed
    try:
        import spotdl
    except ImportError:
        raise ImportError("spotdl is not installed. Please install it using 'pip install spotdl'")

    # Check if the songs directory exists, if not create it
    if not os.path.exists(SONGS_DIR):
      os.makedirs(SONGS_DIR)
      logging.info("Songs directory does not exist. Creating it...")
    
    # Check if the songs directory is empty
    if not os.listdir(SONGS_DIR):
      logging.info("Songs directory is empty. Downloading songs...")

      # Download songs from the playlist URL
      subprocess.run(
        ["spotdl", SPOTIFY_PLAYLIST_URL],
        check=True,
        cwd=SONGS_DIR,
      )
      
      logging.info("Songs downloaded successfully.")
    else:
      logging.info("Songs directory already exists. Skipping download...")