import datetime
import time


class Timer(object):
    """
    A simple timer.
    """

    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.remain_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multi treading
        self.start_time = time.time()
        return self.start_time

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) * \
                               (max_iters - iters) / iters
        return str(datetime.timedelta(seconds=int(self.remain_time)))

    def current_time(self):
        return time.time()

    def initial_time(self):
        return self.init_time

    def avg_time(self):
        return self.average_time

    def elapsed_time(self):
        return self.current_time() - self.initial_time()
