import logging
import os.path
import sys
import time

import psutil

# sys.path.append('/home/ezafati/mesh_project/src/')
import systemutils

path1, *_ = os.path.split(os.path.abspath('./'))

FILE_LOG_PATH = os.path.join(path1, 'src/log/mesh.log')
try:
    logging.basicConfig(filename=FILE_LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
except FileNotFoundError:
    sys.exit(f'FILE {FILE_LOG_PATH} NOT FOUND')

log = logging.getLogger('mesh_log')

#sys.stdout = meshutils.LogWrite(log.info)
#sys.stderr = meshutils.LogWrite(log.error)
sys.excepthook = systemutils.exception_logging


p = psutil.Process()
logging.info(f'Current Process with pid {p.pid} with status {p.status()} launched by the user {p.username()}')
from read_data import read_file

os.nice(20)

print('BEGIN OF THE PROGRAM')
t1 = time.time()
result = read_file("maillage9.txt", p)
if result:
    print('mesh failed')
t2 = time.time()
print(t2-t1)

logging.info(f' Process  {p.pid}  terminates: user CPU time {p.cpu_times().user}, system cpu time {p.cpu_times().system}')
