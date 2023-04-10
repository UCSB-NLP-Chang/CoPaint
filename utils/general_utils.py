import os
import random
import datetime
import numpy as np


def get_random_time_stamp():
    """
    Return a random time stamp.
    :return: random time stamp
    :rtype: str
    """
    return "%d-%s" % (
        random.randint(100, 999),
        datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"),
    )


def makedir_if_not_exist(name):
    """
    Make the directory if it does not exist.
    :param name: dir name
    :type name: str
    """
    if not os.path.exists(name):
        os.makedirs(name)


def random_split(n, ratio=(0.8, 0.1, 0.1)):
    assert sum(ratio) == 1.0
    ret = []
    order = np.random.permutation(n)
    s = 0
    for r in ratio:
        e = s + int(r * n)
        ret.append(order[s:e])
        s = e
    return ret
