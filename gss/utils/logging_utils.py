import logging.config


def configure_logging(log_level, log_path=None):
    handlers = {
        "out": {
            "class": "logging.StreamHandler",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        }
    }
    if log_path is not None:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "formatter": "basic",
            "filename": log_path,
            "mode": "w",
            "encoding": "utf-8",
        }
    CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "basic": {
                "format": "%(asctime)s %(name)s %(pathname)s:%(lineno)d - %(levelname)s - %(message)s"
            }
        },
        "handlers": handlers,
        "loggers": {"gss": {"handlers": handlers.keys(), "level": log_level}},
        "root": {"handlers": handlers.keys(), "level": log_level},
    }
    logging.config.dictConfig(CONFIG)


def get_logger():
    return logging.getLogger("gss")
