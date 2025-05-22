import logging
import coloredlogs

logger = logging.getLogger(__name__)

coloredlogs.install(
  level='INFO', 
  logger=logger, 
  fmt='%(asctime)s [%(levelname)s] %(message)s', 
  datefmt='%I:%M:%S %p', 
  isatty=True
)
