import logging
import sys

_console='console'
_file='file'

HCONSOLE = {'handler':_console}
HFILE = {'handler':_file}


logger = logging.getLogger("test")
logger.setLevel(level=logging.DEBUG)

logStreamFormatter = logging.Formatter(
  fmt=f"%(levelname)-8s %(asctime)s \t %(filename)s %(funcName)s line %(lineno)s - %(message)s", 
  datefmt="%H:%M:%S"
)

consoleHandler = logging.StreamHandler(stream=sys.stdout)
consoleHandler.setFormatter(logStreamFormatter)
consoleHandler.setLevel(level=logging.DEBUG)
consoleHandler.name=_console

logger.addHandler(consoleHandler)

logFileFormatter = logging.Formatter(
    fmt=f"%(levelname)s %(asctime)s   -->  %(filename)s F-%(funcName)s L-%(lineno)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
fileHandler = logging.FileHandler(filename='log.log',mode="a")
fileHandler.setFormatter(logFileFormatter)
fileHandler.setLevel(level=logging.INFO)
fileHandler.name=_file

logger.addHandler(fileHandler)