import multiprocessing
import sys
import __main__
from pathlib import Path

from loguru import logger


def get_logger(enable):
    # TensorRT logging is handled separately inside the Runner classes

    multiprocessing.current_process().name = Path(__main__.__file__).stem

    logger.remove()

    if enable:
        logger.add(sys.stdout,
                   format="<fg magenta>{time:YYYY-MM-DD HH:mm:ss:SSS ZZ}</> <yellow>|</>"
                          " <lvl>{level: <8}</> "
                          "<yellow>|</> <blue>{process.name: ^12}</> <yellow>-</> <lvl>{message}</>",
                   diagnose=True)  # b28774 (magenta)

        logger.level('INFO', color='<fg white>')  # fef5ed
        logger.level('SUCCESS', color='<fg green>')  # 79d70f
        logger.level('WARNING', color='<fg yellow>')  # fd811e
        logger.level('ERROR', color='<fg red>')  # ed254e

    return logger


if __name__ == '__main__':
    logger = setup_logging(True)
    logger.success('Ciao')
