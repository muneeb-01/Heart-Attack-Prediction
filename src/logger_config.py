# logger_config.py
import logging
import os
from datetime import datetime

# Create log directory if it doesn’t exist
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Define log file name
LOG_FILE = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging only once
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d - %(name)s - %(message)s"
)

# Create a module-specific logger
logger = logging.getLogger(__name__)
