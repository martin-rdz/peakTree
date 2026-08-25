#! /usr/bin/env python3
# coding=utf-8
""""""
"""
Author: radenz@tropos.de
"""

import datetime
from pathlib import Path
import logging
import subprocess
import netCDF4
import numpy as np
from . import helpers as h
from . import generate_tree
import toml

from typing import Union

import xarray as xr    
from rpgpy import read_rpg

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(levelname)s: %(message)s')
# stream_handler.setFormatter(formatter)
# file_handler = logging.FileHandler(filename='../test.log', mode='w')
# formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s', datefmt='%H:%M:%S')
# file_handler.setFormatter(formatter)
# file_handler.setLevel(logging.DEBUG)
# log.addHandler(file_handler)
# log.addHandler(stream_handler)


from ._meta import __version__, __author__

#@profile
def check_part_not_reproduced(tree, spectrum):
    """check how good the moments in the tree (only leave nodes)
    represent the original spectrum (i.e. if there are non-Gaussian peaks)
    
    Args:
        tree: a tree in the traversed (dict) format
        spectrum: and the corresponding spectrum

    Returns:
        number of bins, where the reprocduced spectrum differs by more than 7dB
    """
    parents = [n.get('parent_id', -1) for n in tree.values()]
    leave_ids = list(set(tree.keys()) - set(parents))
    spec_from_mom = np.zeros(spectrum['specZ'].shape)
    vel, vel_mask = h.masked_to_plain(spectrum['vel'])
    if 'vel_step' not in spectrum:
        delta_v = vel[~vel_mask][2] - vel[~vel_mask][1]
    else:
        delta_v = spectrum['vel_step']
    
    for i in leave_ids:
        if tree[i]['width'] < 0.001:
            tree[i]['width'] = 0.0001
        S = tree[i]['z'] * delta_v
        # calculate Gaussian only in a small range
        ivmean = np.searchsorted(spectrum['vel'], tree[i]['v'])
        step = int(7*tree[i]['width']/delta_v)
        ista, iend = ivmean - step, ivmean + step
        spec_from_mom[ista:iend] += S * h.gauss_func(spectrum['vel'][ista:iend], tree[i]['v'], tree[i]['width'])
        
    spec_from_mom[spec_from_mom < spectrum['noise_thres']] = spectrum['noise_thres']
    difference = spectrum['specZ']/spec_from_mom
   
    return np.count_nonzero(np.abs(difference[~spectrum['specZ_mask']]) > h.z2lin(7))*delta_v



def get_git_hash():
    """
    Returns:
        git describe string
    """
    try:
        commit = subprocess.check_output(['git', 'describe', '--always'])
        branch = subprocess.check_output(['git', 'branch', '--show-current'])
    except:
        commit = 'git error'
        branch = 'git error'
        log.warning(commit)
    return commit.rstrip(), branch.rstrip()



def get_time_grid(timestamps, ts_range, time_interval, filter_empty=True):
    """get the mapping from timestamp indices to gridded times
    eg for use in interpolation routines

    https://gist.github.com/martin-rdz/b7c3b9f06bb41aeb6b2fb6c888275e26
    
    Args:
        timestamps: list of timestamps
        ts_range: range fo the gridded timestamps
        time_interval: interval of the gridded timestamps
        filter_empty (bool, optional): include the bins that are empty
    Returns:
        list of (timestamp_begin, timestamp_end, grid_mid, index_begin, index_end, no_indices)
    """
    print('get_time_grid ', ts_range[0], ts_range[1])
    grid = np.arange(ts_range[0], ts_range[1]+1, time_interval)
    grid_mid = grid[:-1] + np.diff(grid)/2

    corresponding_grid = np.digitize(timestamps, grid)-1
    bincount = np.bincount(corresponding_grid)
    end_index = np.cumsum(bincount)
    begin_index = end_index - bincount
    
    out = zip(grid[:-1], grid[1:], grid_mid, begin_index, end_index, bincount)
    if filter_empty:
        out = filter(lambda x: x[5] !=0, out)
    out = list(out)
    return [np.array(list(map(lambda e: e[i], out))) for i in range(6)]


def get_averaging_boundaries(array, slice_length, zero_index=0):
    """get the left and right indices each element in an array
    for a given averaging slice_length
    """

    is_left = np.digitize(array-slice_length/2., array)
    is_right = np.digitize(array+slice_length/2., array, right=True)

    #print(is_left[0], is_right[0])
    #print(array[is_left[0]], array[0], array[is_right[0]])

    return zero_index + is_left, zero_index + is_right

    
def roll_velocity_dataset(ds, config):
    """roll the spectra, i.e., glue the rightmost x m/s to the left
    
    """
    
    vel_step = (ds.doppler[1] - ds.doppler[0]).values
    bin_roll_velocity = (config['roll_velocity']/vel_step).astype(int)
    print(bin_roll_velocity, bin_roll_velocity*vel_step)

    ds_rolled = ds.roll(doppler=bin_roll_velocity)
    ds_rolled.coords['doppler'] = ds.coords['doppler'] - bin_roll_velocity*vel_step

    return ds_rolled
    


def load_rpgbinary(filename):
    """load the rpg LV0 files into a xr.datatree structure
    
    .. TODO::
        maybe track the chirp boundaries also in output file
    """

    header, data = read_rpg(filename)
    offset = (datetime.datetime(2001,1,1) - datetime.datetime(1970, 1, 1)).total_seconds()
    ts = offset + data['Time'] + data['MSec']*1e-3
    
    # inheriting the time probably does not make sense if we want to do individual resampling
    datatree = xr.DataTree(name='root')
    
    rg = header['RAlts']
    chirp_start_indices = header['RngOffs']
    no_chirps = chirp_start_indices.shape[0]
    #print(f'chirp_start_indices {chirp_start_indices}')
    bins_per_chirp = np.diff(np.hstack((chirp_start_indices, rg.shape[0])))
    #print(f'range bins per chirp {bins_per_chirp} {bins_per_chirp.shape}')
    specN = header['SpecN']
    #print('SpecN', specN)
    
    velocity_vector = header['velocity_vectors']
    
    rg_chirp_map = np.repeat(np.arange(no_chirps), bins_per_chirp)

    for sel_chirp in range(no_chirps):

        specN_half = int(specN[sel_chirp]/2)
        spec_midpoint = int(data['TotSpec'].shape[2]/2)

        specslice = slice(spec_midpoint-specN_half, spec_midpoint+specN_half)
        print('specslice', specslice)
        # ignore the scaling for now
        spec_tot = data['TotSpec'][:,rg_chirp_map == sel_chirp, specslice]
        spec_h = data['HSpec'][:,rg_chirp_map == sel_chirp, specslice]
        spec_cov_re = data['ReVHSpec'][:,rg_chirp_map == sel_chirp, specslice]
        spec_cov_im = data['ImVHSpec'][:,rg_chirp_map == sel_chirp, specslice]
        spec_v = 4 * spec_tot - spec_h - 2 * spec_cov_re
        # adding the noise omitted here
        noise_v = data['TotNoisePow'][:,rg_chirp_map == sel_chirp]/specN[sel_chirp]
        noise_h = data['HNoisePow'][:,rg_chirp_map == sel_chirp]/specN[sel_chirp]

        spec_h += noise_h[...,np.newaxis]
        spec_v += noise_v[...,np.newaxis]
        noise_combined = (noise_v + noise_h) / 2

        # quality filters suggested by Alexander
        mask = (spec_tot < 1.1e-10) | (spec_h < 1.1e-10)

        rhv = np.sqrt(spec_cov_re**2 + spec_cov_im**2) / np.sqrt(spec_v * spec_h)

        Z = (spec_v + spec_h)*(1+rhv) / 2 - noise_combined[...,np.newaxis]
        Zcx = (spec_v + spec_h)*(1-rhv) / 2 - noise_combined[...,np.newaxis]
        print(Z.shape)

        Z[np.isnan(Z)] = 0
        Z[mask] = 0
        Zcx[np.isnan(Zcx)] = 0

        ds = xr.Dataset(
            data_vars = dict(
                Z=(('time', 'range', 'doppler'), Z),
                Zcx=(('time', 'range', 'doppler'), Zcx),
                noise=(('time', 'range'), noise_combined),
                ),
            coords=dict(
                time=('time', ts.astype('datetime64[s]')),
                range=('range', rg[rg_chirp_map == sel_chirp]),
                doppler=('doppler', velocity_vector[sel_chirp,specslice]),
                ),
            #attrs=dict(),
        )
        datatree[f"chirp{sel_chirp+1}"] = xr.DataTree(ds)

    return datatree


def load_znc(filename):
    """load the Metek MIRA znc spectra into a xr.Dataset
    
     
    """


    ds = xr.open_dataset(filename)
    ds['time'] = (ds['time']*1e6 + ds['microsec']).astype('datetime64[us]')

    Z = ds['SPCco'] / ds['npw1'] * ds['RadarConst'] * (ds['range'] / 5e3)**2 * ds['SNRCorFaCo']
    Z.attrs['long_name'] = 'Spectral reflectivity'
    
    nfft = ds['doppler'].values.shape[0]
    noise = ds['HSDco'] * ds['RadarConst'] * (ds['range'] / 5e3)**2 / nfft 
    
    Zcx = ds['SPCcx'] / ds['npw2'] * ds['RadarConst'] * (ds['range'] / 5e3)**2 * ds['SNRCorFaCx']
    
    no_roll = int(ds['doppler'].shape[0]/2)
    print('no_roll', no_roll)
    Z = Z.roll(doppler=no_roll, roll_coords=True).isel(doppler=slice(None, None, -1))
    Zcx = Zcx.roll(doppler=no_roll, roll_coords=True).isel(doppler=slice(None, None, -1))
    Z['doppler'] = Z['doppler']*-1
    Zcx['doppler'] = Zcx['doppler']*-1
    
    ds_input = xr.Dataset(
        data_vars={
            "Z": Z,
            "Zcx": Zcx,
            "noise": noise,
        }
    )

    return ds_input


def check_and_nest_parameters(keys, d):
    """Check and nest parameters based on provided keys.

    Parameters
    ----------
    keys : set
        Expected parameter keys.
    d : dict
        Dictionary containing parameters.

    Returns
    -------
    dict
        Dictionary with parameters nested according to keys.

    Raises
    ------
    ValueError
        If tree_params does not contain sufficient keys.
    """

    if set(keys) == set(d.keys()):
        # names of datatree nodes are in tree_params
        print('all keys there, nothing to do', list(keys))
        return d
    elif set(d.keys()).issubset(set(keys)):
        raise ValueError('tree_params not sufficient keys')
    else:
        return {k: d for k in keys}

        
def map_over_dataset_nested_args(datatree, func, *args):
    """custom version of xr.map_over_datasets, which allows for nested args,configs

        

    
    .. code:: python
    
        args = ({'resample_time': '6s'},)
        # is expanded to 
        [{'chirp1': {'resample_time': '6s'}, 'chirp2': {'resample_time': '6s'}, 'chirp3': {'resample_time': '6s'}}]
    
    """

    print('args ', args)
    args = [check_and_nest_parameters(datatree.keys(), a) for a in args]
    print('nested args ', args)

    d = {}
    for path, node in datatree.subtree_with_keys:
        if not node.has_data:
            continue
        #print(path, node)
        d[path] = func(node.dataset, *[e[path] for e in args])

    return xr.DataTree.from_dict(d)


def to_tree(data: Union[xr.Dataset, xr.DataTree], params: dict, meta: dict):
    """ 
    
    Notes
    -----
    
    `xr.map_over_datasets` does not allow for per node function parameters
    
    
    """
    
    if isinstance(data, xr.core.dataset.Dataset):
        return ds_to_tree(data, params, meta)

    # just for now, might remove later
    assert isinstance(data, xr.core.datatree.DataTree)

    #params = check_and_nest_parameters(data.keys(), params)
    #meta = check_and_nest_parameters(data.keys(), meta)

    return map_over_dataset_nested_args(data, ds_to_tree, params, meta)



def ds_to_tree(ds_input: xr.Dataset, params: dict, meta: dict):
    """Convert an xarray dataset of spectra into a peakTree dataset.

    Parameters
    ----------
    ds_input : xarray.Dataset
        Input dataset containing one or more variables along the ``inputvar``
        and ``doppler`` dimensions, plus a boolean ``noise_mask``.
    params : dict
        Parameters passed to the tree-building routine, such as width and
        prominence thresholds.
    meta : dict
        Mapping from variable names to the moment names used when computing
        moments.

    Returns
    -------
    xarray.Dataset
        Rectangularized tree dataset with node-wise bounds, parent/child
        relationships, and moment variables.

    Notes
    -----
    The implementation applies :func:`generate_tree.ufunc_wrapper` to each
    spectrum in the input dataset via :func:`xarray.apply_ufunc`.

    Examples
    --------
    >>> meta = {
    ...     'Z': ['M0', 'M1', 'M2', 'M3', 'P'],
    ...     'Zcx': ['M0', 'P'],
    ... }
    >>> ds_input = xr.Dataset(
    ...     data_vars={
    ...         "Z": Z,
    ...         "Zcx": Zcx,
    ...         "noise_mask": Z < noise * 1.3,
    ...     }
    ... )
    >>> params = {'width_thres': 0.1, 'prom_thres': 1}
    ...
    >>> ds_rect = peakTree.ds_to_tree(
    ...     ds_input, params, meta)


    """

    ds_input_array = ds_input.to_array(dim="inputvar")
    vel_step = (ds_input['doppler'][1] - ds_input['doppler'][0]).values

    ds1, ds2 = xr.apply_ufunc(
        generate_tree.ufunc_wrapper,
        ds_input_array.doppler,
        ds_input_array.inputvar,
        ds_input_array,
        ds_input['noise_mask'],
        kwargs={
            'vel_step':vel_step, 
            'params': params,
            'var_peak': 'Z',
            'meta': meta },
        input_core_dims=[['doppler'], ['inputvar'], ['inputvar', 'doppler'], ['doppler']],
        output_core_dims=[['var', 'node'], ['var']],
        vectorize=True
    )

    ds1.coords['var'] = ds2.isel(range=0, time=0).values
    ds_rect = ds1.to_dataset(dim='var')
    ds_rect['no_nodes'] = (ds_rect['bounds_left'] != -999).sum(dim='node')

    for v in ['bounds_left', 'bounds_right', 'id_parent']:
        ds_rect[v] = ds_rect[v].astype('int32')
    node_ids = ds_rect.coords['node'].values
    ds_rect['id_child_left'] = ('node', 2*node_ids + 1)
    ds_rect['id_child_right'] = ('node', 2*node_ids + 2)
    ds_rect['has_children'] = xr.apply_ufunc(
        h.has_children,
        ds_rect.id_parent,
        input_core_dims=[['node']],
        output_core_dims=[['node']],
        vectorize=True
    )
    ds_rect['is_leaf'] = xr.where(
        (ds_rect.id_parent != -999) & ~ds_rect.has_children, True, False)
    return ds_rect

    
    
def store_to_netcdf(
        ds, savepath, short_location='', system_name='',
        contact='', institution=''):
    """enrich the xr.Dataset with some metadata and store to netcdf
    
    try to mimic the information als present in the original output files

    
    .. TODO::
        The original version stores the velocity arrays, but that information is hard to map 
        (especially when chirps are present). Maybe its better to store the velocity for each node directly.

    .. TODO::
        location, inputinfo, location, commit and branch are missing

    .. TODO::
        cloudnet hours and true unix timestamps might also be needed
        
        
    .. TODO::
        Think of optimizing variable sorting

    .. TODO::
        not working for datatrees yet because of the time, but for rpg stacking might be an option
        also datatree.to_netcdf is incompatible with path=
    
    """

    begin_dt = ds.time.values[0].astype('datetime64[us]').astype('O')

    
    ds.attrs['description'] = 'peakTree processing'
    ds.attrs['software_version'] = __version__
    ds.attrs['day'] = str(begin_dt.day)
    ds.attrs['month'] = str(begin_dt.month)
    ds.attrs['year'] = str(begin_dt.year)
    ds.attrs['contact'] = contact
    ds.attrs['institution'] = institution
    ds.attrs['creation_time'] = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    
    savefile=Path(savepath) / f'{begin_dt:%Y%m%d_%H%M}_{short_location}_{system_name}_peakTree.nc4'
    ds.to_netcdf(
        path=savefile
    )
    print('saved to ', savefile)
    
