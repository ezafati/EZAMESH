import logging
import sys, os.path
import time

import psutil
import meshutils
# sys.path.append('/home/ezafati/mesh_project/src/')

path1, *_ = os.path.split(os.path.abspath('./'))

FILE_LOG_PATH = os.path.join(path1, 'src/log/mesh.log')
try:
    logging.basicConfig(filename=FILE_LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
except FileNotFoundError:
    sys.exit(f'FILE {FILE_LOG_PATH} NOT FOUND')

log = logging.getLogger('mesh_log')

#sys.stdout = meshutils.LogWrite(log.info)
#sys.stderr = meshutils.LogWrite(log.error)
#sys.excepthook = meshutils.exception_logging


p = psutil.Process()
logging.info(f'Current Process with pid {p.pid} with status {p.status()} launched by the user {p.username()}')
from read_data import *

print('BEGIN OF THE PROGRAM')
t1 = time.time()
read_file("maillage5.txt", p)
t2 = time.time()
print(t2-t1)

logging.info(f' Process  {p.pid}  terminates: user CPU time {p.cpu_times().user}, system cpu time {p.cpu_times().system}')
