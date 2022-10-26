import multiprocessing
import sys
import __main__
from pathlib import Path

from loguru import logger


# TensorRT logging is handled separately inside the Runner classes
def setup_logger(level=0, name=None):
    """level: set the minimum severity level to be printed out
        name: sets the name displayed in the error messages
        disable: makes all logger calls no-op
    """

    if isinstance(level, list):
        lvl_filter = lambda msg: msg['level'].no in level
    else:
        lvl_filter = lambda msg: msg['level'].no >= level

    if name is None:
        multiprocessing.current_process().name = Path(__main__.__file__).stem

    logger.remove()

    logger.add(sys.stdout,
               format="<fg magenta>{time:YYYY-MM-DD HH:mm:ss:SSS ZZ}</> <yellow>|</>"
                      " <lvl>{level: <8}</> "
                      "<yellow>|</> <blue>{process.name: ^12}</> <yellow>-</> <lvl>{message}</>",
               diagnose=True, filter=lvl_filter)  # b28774 (magenta)

    logger.level('INFO', color='<fg white>')  # fef5ed
    logger.level('SUCCESS', color='<fg green>')  # 79d70f
    logger.level('WARNING', color='<fg yellow>')  # fd811e
    logger.level('ERROR', color='<fg red>')  # ed254e

    return


# if __name__ == '__main__':
#     setup_logger()
#     logger.success('Ciao')
