import logging
import sys
import colorlog

def get_logger(name=None):
    logger = colorlog.getLogger(name or "Baseline")

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler(stream=sys.stdout)

    formatter = colorlog.ColoredFormatter(
        '%(asctime_log_color)s[%(asctime)s] '
        '%(levelname_log_color)s[%(levelname)s] '
        '%(name_log_color)s[%(name)s] '
        '%(message_log_color)s- %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={
            'asctime': {
                'DEBUG': 'white',
                'INFO': 'cyan',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            'levelname': {
                'DEBUG': 'white',
                'INFO': 'yellow',
                'WARNING': 'blue',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            'name': {
                'DEBUG': 'white',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            'message': {
                'DEBUG': 'green',
                'INFO': 'white',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
        },
        style='%'
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger