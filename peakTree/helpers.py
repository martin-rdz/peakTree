#! /usr/bin/env python3
# coding=utf-8
"""
Collection of common helper functions.
"""
"""
Author: radenz@tropos.de
"""

import datetime
import numpy as np


def list_of_elem(elem, length):
    """return a list of given length of given elements"""
    return [elem for i in range(length)]


def epoch_to_timestamp(time_raw):
    """converts raw time (days since year 0) to unix timestamp
    """
    offset = 719529  # offset between 1970-1-1 und 0000-1-1
    time = (time_raw - offset) * 86400
    return time


def dt_to_ts(dt):
    """convert a datetime to unix timestamp"""
    # timestamp_midnight = int((datetime.datetime(self.dt[1].year, self.dt[1].month, self.dt[1].day) - datetime.datetime(1970, 1, 1)) / datetime.timedelta(seconds=1)) #python3
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()


def ts_to_dt(ts):
    """convert a unix timestamp to datetime"""
    return datetime.datetime.utcfromtimestamp(ts)


def lin2z(array):
    """calculate dB values from a array of linear"""
    return 10*np.log10(array)


def z2lin(array):
    """calculate linear values from a array of dBs"""
    return 10**(array/10.)

def fill_with(array, mask, fill):
    """fill array with fill value where mask is True"""
    filled = array.copy()
    filled[mask] = fill
    return filled