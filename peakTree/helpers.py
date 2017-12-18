#! /usr/bin/env python3
# coding=utf-8
"""
Author: radenz@tropos.de

collection of tiny helper functions

"""

import datetime
import numpy as np


def list_of_elem(elem, length):
    return [elem for i in range(length)]


def epoch_to_timestamp(time_raw):
    """
    converts rata die (days since year 0) to unix timestamp
    """
    offset = 719529  # offset between 1970-1-1 und 0000-1-1
    time = (time_raw - offset) * 86400
    return time


def dt_to_ts(dt):
    # timestamp_midnight = int((datetime.datetime(self.dt[1].year, self.dt[1].month, self.dt[1].day) - datetime.datetime(1970, 1, 1)) / datetime.timedelta(seconds=1)) #python3
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()


def ts_to_dt(ts):
    return datetime.datetime.utcfromtimestamp(ts)


def lin2z(array):
    return 10*np.log10(array)


def z2lin(array):
    return 10**(array/10.)

def fill_with(array, mask, fill):
    filled = array.copy()
    filled[mask] = fill
    return filled