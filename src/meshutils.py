import logging
import traceback

from contextlib import contextmanager
from multiprocessing import Process
import multiprocessing


class LogWrite:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        pass


def exception_logging(exctype, value, tb):
    write_val = {'exception_type': str(exctype),
                 'message': str(traceback.format_tb(tb, 10))}
    logging.exception(str(write_val))


class TaskProcess:
    def __init__(self, tr=None, fcond=None, args=None):
        self.root_tr = tr
        self.func = fcond
        self.extra = args


@contextmanager
def launch_processes(task_queue, *args):
    npr = multiprocessing.cpu_count()
    try:
        for _ in range(npr):
            Process(target=worker_test, args=(task_queue, *args)).start()
        yield
    finally:
        for _ in range(npr):
            task_queue.put('STOP')


def worker_test(task_que, *args):
    for func, kwargs in iter(task_que.get, 'STOP'):
        func(task_que, *args, **kwargs)
