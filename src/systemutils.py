import logging
import sys
import traceback

from contextlib import contextmanager
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
                 'message': str(traceback.format_tb(tb, 10)) + str(value)}
    logging.exception(str(write_val))
    sys.exit('MESH CREATION FAILED: SEE LOG FILE')


class NotApproValueError(Exception):
    """error raised when the input value is not
    appropriate """


class AlreadyExistError(Exception):
    """error raised when the object is already
     provided with the same label for instance"""


class UnknownElementError(Exception):
    """error raised when the provided element
    is unkown or undefined """


@contextmanager
def launch_processes(task_queue, func, *args):
    npr = multiprocessing.cpu_count()
    try:
        for _ in range(npr):
            multiprocessing.Process(target=func, args=(task_queue, *args)).start()
        yield
    finally:
        for _ in range(npr):
            task_queue.put('STOP')


def worker(task_que, *args):
    for func, kwargs in iter(task_que.get, 'STOP'):
        func(task_que, *args, **kwargs)
