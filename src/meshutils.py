import logging
import traceback


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
