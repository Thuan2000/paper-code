import sys
import logging
from logging.config import fileConfig
from logging.handlers import RotatingFileHandler
from config import Config
from core.cv_utils import create_if_not_exist


# add checking before run
create_if_not_exist(Config.Dir.DATA_DIR)
create_if_not_exist(Config.Dir.LOG_DIR)

logger = logging.getLogger(Config.LogFile.LOG_NAME)
if Config.Mode.PRODUCTION:
    handler = RotatingFileHandler(Config.LogFile.LOG_FILE)
else:
    handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt='[%(asctime)s] [%(levelname)s] %(name)s - %(processName)s: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
