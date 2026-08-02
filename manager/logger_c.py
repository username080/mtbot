import logging
from logging.handlers import TimedRotatingFileHandler
import os

class QuantLogger:
    def __init__(self, log_dir="logs", log_file="quant_system.log", level=logging.INFO):
        self.logger = logging.getLogger("QuantSystem")
        self.logger.setLevel(level)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_path = os.path.join(log_dir, log_file)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        handler = TimedRotatingFileHandler(
            filename=log_path,
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8"
        )

        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger
    
