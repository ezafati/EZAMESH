import logging
import sys

import psutil

# sys.path.append('/home/ezafati/mesh_project/src/')
FILE_LOG_PATH = 'mesh.log'
try:
    logging.basicConfig(filename=FILE_LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
except FileNotFoundError:
    sys.exit(f'FILE {FILE_LOG_PATH} NOT FOUND')

p = psutil.Process()
logging.info(f'Current Process with pid {p.pid} with status {p.status()} launched by the user {p.username()}')
from read_data import *

read_file("maillage5.txt", p)

logging.info(f' Process  {p.pid}  terminates: user CPU time {p.cpu_times().user}, system cpu time {p.cpu_times().system}')
