import logging
import sys, os.path

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

p = psutil.Process()
logging.info(f'Current Process with pid {p.pid} with status {p.status()} launched by the user {p.username()}')
from read_data import *

print('BEGIN OF THE PROGRAM')
read_file("maillage2.txt", p)

logging.info(f' Process  {p.pid}  terminates: user CPU time {p.cpu_times().user}, system cpu time {p.cpu_times().system}')
