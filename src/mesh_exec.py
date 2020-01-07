#!/usr/bin/python3
import argparse
import logging
import os
import sys
import time
from read_data import read_file

import psutil

import systemutils

parser = argparse.ArgumentParser()
parser.add_argument("meshfile", help='The input mesh file', type=str)
args = parser.parse_args()

# configure logging
FILE_LOG_PATH = dict(os.environ)['MESH_LOG_PATH']
try:
    logging.basicConfig(filename=FILE_LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
except FileNotFoundError:
    sys.exit(f'FILE {FILE_LOG_PATH} NOT FOUND')

# redirect the stdout and stderr to log file
log = logging.getLogger('mesh_log')
# sys.stdout = systemutils.LogWrite(log.info)
# sys.stderr = systemutils.LogWrite(log.error)
sys.excepthook = systemutils.exception_logging

# launch the program
p = psutil.Process()  # to keep track of some system properties
logging.info(f'Current Process with pid {p.pid} with status {p.status()} launched by the user {p.username()}')
print('BEGIN OF THE PROGRAM')
t1 = time.time()
result = read_file(args.meshfile, p)
t2 = time.time()
print('End of the PROGRAM  with running time', t2 - t1)
logging.info(
    f' Process  {p.pid}  terminates: user CPU time {p.cpu_times().user}, system cpu time {p.cpu_times().system}')
