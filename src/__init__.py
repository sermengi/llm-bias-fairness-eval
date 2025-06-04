import logging
import os
import sys
from datetime import datetime

LOG_FILE = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

log_formatter = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

logging.basicConfig(
    level=logging.INFO,
    format=log_formatter,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("llmevallogger")
