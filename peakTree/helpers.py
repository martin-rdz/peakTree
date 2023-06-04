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
import scipy
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


def masked_to_plain(array):
    """unpack the masked array into plain versions"""
    if isinstance(array, np.ma.MaskedArray):
        mask = array.mask
        mask = mask if mask.shape == array.data.shape else np.zeros_like(array).astype(bool)
        return array.data, mask
    else:
        return array, np.zeros_like(array).astype(bool)


def divide_bounds(bounds):
    """
    divide_bounds([[10,20],[20,25],[25,30],[40,50]]) 
    => noise_sep [[10, 30], [40, 50]], internal [20, 25]
    """
    bounds = list(sorted(flatten(bounds)))
    occ = dict((k, (bounds).count(k)) for k in set(bounds))
    internal = [k for k in occ if occ[k] == 2]
    noise_sep = [b for b in bounds if b not in internal]
    noise_sep = [[noise_sep[i], noise_sep[i+1]] for i in \
                    range(0, len(noise_sep)-1,2)]
    return noise_sep, internal

@jit(nopython=True, fastmath=True)
def lin2z(array):
    """calculate dB values from a array of linear"""
    return 10*np.log10(array)


@jit(nopython=True, fastmath=True)
def z2lin(array):
    """calculate linear values from a array of dBs"""
    return 10**(array/10.)

def fill_with(array, mask, fill):
    """fill array with fill value where mask is True"""
    filled = array.copy()
    filled[mask] = fill
    return filled


def round_odd_old(f):
    return int(np.ceil(f/2.) * 2 + 1)


def round_odd(f):
    """round to odd number
    :param f: float number to be rounded to odd number
    """
    return round(f) if round(f) % 2 == 1 else round(f) + 1

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

def filter_none_rec(e):
    """filter None values from nested list"""
    return list(filter(None, [filter_none_rec(y) for y in e])) if isinstance(e, list) else e

@jit(nopython=True, fastmath=True)
def gauss_func(x, m, sd):
    """calculate the gaussian function on a given grid
    Args:
        x (array): grid
        m (float): mean
        sd (float): standard deviation
    Returns:
        array
    """
    a = 1. / (sd * np.sqrt(2. * np.pi))
    return a * np.exp(-(x - m) ** 2 / (2. * sd ** 2))

# slightly different formulation for the tail filter
@jit(nopython=True, fastmath=True)
def gauss_func_offset(x, H, A, x0, sigma):
    """x, H, A, x0, sigma"""
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    #print('init guess ', [min(y), max(y)-min(y), mean, sigma])
    popt, pcov = scipy.optimize.curve_fit(
        gauss_func_offset, x, y, 
        p0=[min(y), max(y)-min(y), mean, sigma],
        #bounds=[(-57, 9, -4, 0.4), (-35, 18, 4, 2)],
        bounds=[(-np.inf, 9, -np.inf, -np.inf), (np.inf, 18, np.inf, np.inf)]
        )
    return popt, pcov


@jit(nopython=False, fastmath=True)
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

@jit(nopython=True, fastmath=True)
def estimate_mean_noise(spec, mov_avg=1):
    i_noise = len(spec)
    spec_sort = np.sort(spec)
    for i in range(spec_sort.shape[0]):
        partial = spec_sort[:i+1]
        mean = partial.mean()
        var = partial.var()
        if var * mov_avg * 2 < mean**2.:
            i_noise = i
        else:
            break
    noise_part = spec_sort[:i_noise+1]
    return np.mean(noise_part)


def estimate_noise_array(spectra_array):
    """
    Wrapper for estimate_noise, to apply to a chunk of Doppler spectra
    Args:
         spectra_array (ndarray): 3D array of Doppler spectra
    Returns:
         mean noise for each time-height
    """
    print('estimating noise...')
    out = np.zeros(spectra_array.shape[:2])-999.0
    for ts in range(spectra_array.shape[0]):
        for rg in range(spectra_array.shape[1]):
            out[ts, rg] = estimate_mean_noise(spectra_array[ts, rg, :])

    return out


def blur_mask(msk, width, axis=-1):
    """Reduce the thresholding effect for spectra with single bins above the estimated noise level
    
    should mimic blurthres=1 or 2 in spectraprocessing
    
    Example:
        ```
        msk = (Zco[it,ir] < 1.01*noiseCo[it,ir])

        fig, ax = plt.subplots(1, figsize=(9, 5))
        ax.plot(msk, label='wo blur')
        ax.plot(
            blur_mask(msk, 5),
            label='blured',)

        ax.legend()
        ax.set_xlim(-3, 100)
        ax.set_ylabel('Mask 0=valid 1=noise')
        ax.set_xlabel('Bin')
        ```
    """
    msk_wide = scipy.ndimage.convolve1d(
        (msk).astype(float), np.ones(width)/width, mode='wrap', axis=axis)
    print('mask_blur', np.sum(msk), np.sum(msk_wide > 0.8))
    return msk_wide > 0.8