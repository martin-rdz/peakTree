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
from numba import jit


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


def flatten(xs):
    """flatten inhomogeneous deep lists
    e.g. ``[[1,2,3],4,5,[6,[7,8],9],10]``
    """
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten(x))
    else:
        result.append(xs)
    return result


@jit
def estimate_noise(spec, mov_avg=1):
    """
    Noise estimate based on Hildebrand and Sekhon (1974)
    """
    i_noise = len(spec)
    spec_sort = np.sort(spec)
    for i in range(spec_sort.shape[0]):
        partial = spec_sort[:i+1]
        mean = partial.mean()
        var = partial.var()
        if var * mov_avg * 2 < mean**2.:
            i_noise = i
        else:
            # remaining part of spectrum has no noise characteristic
            break
    noise_part = spec_sort[:i_noise+1]
    # case no signal above noise
    noise_sep = spec_sort[i_noise] if i_noise < spec.shape[0] else np.mean(noise_part)

    return {'noise_mean': np.mean(noise_part), 
            'noise_sep': noise_sep,
            'noise_var': np.var(noise_part), 
            'no_noise_bins': i_noise}