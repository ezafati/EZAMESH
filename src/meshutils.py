class LogWrite:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message != '\n':
            self.level(message)

    def flush(self):
        pass


class TaskProcess:
    def __init__(self, tr=None, fcond=None, args=None):
        self.root_tr = tr
        self.func = fcond
        self.extra = args
