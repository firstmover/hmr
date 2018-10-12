from time import time
import numpy as np


class Timer:
    """ timer """

    silent = False
    memo = {}
    hierarchy = []

    def __init__(self, namespace):
        Timer.hierarchy.append(namespace)
        self.name = " / ".join(Timer.hierarchy)

    def __enter__(self):
        return
        self.tic = time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        return 
        toc = time()
        if not Timer.silent:
            print("{} : {}".format(self.name, toc - self.tic))
        if self.name in Timer.memo.keys():
            Timer.memo[self.name].append(toc - self.tic)
        else:
            Timer.memo[self.name] = [toc - self.tic]
        Timer.hierarchy.pop()

    @staticmethod
    def report_memo_average():
        keys = sorted(list(Timer.memo.keys()))
        for k in keys:
            v = Timer.memo[k]
            print("{} : {}".format(k, np.mean(v)))

    @staticmethod
    def clear_memo():
        Timer.memo = {}

    @staticmethod
    def set_silent():
        Timer.silent = True

    @staticmethod
    def set_verbose():
        Timer.silent = False
