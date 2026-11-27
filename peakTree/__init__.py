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
import gc
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


    
def roll_velocity_dataset(ds, config):
    """roll the spectra, i.e., glue the rightmost x m/s to the left
    
    """
    
    vel_step = (ds.doppler[1] - ds.doppler[0]).values
    bin_roll_velocity = (config['roll_velocity']/vel_step).astype(int)
    print(bin_roll_velocity, bin_roll_velocity*vel_step)

    ds_rolled = ds.roll(doppler=bin_roll_velocity)
    ds_rolled.coords['doppler'] = ds.coords['doppler'] - bin_roll_velocity*vel_step

    return ds_rolled


def smooth_doppler(Z, noise, window=5, kernel='gauss', domain='log', fill=1.0,
                   polyorder=2, sigma=None):
    """smooth noise-subtracted spectra along the velocity dimension (``doppler``)

    With ``domain='log'``, ``log(Z + fill*noise)`` is smoothed and ``fill*noise``
    is subtracted afterwards (i.e. smoothing in dB realm). 

    A bit experimentally
    - ``fill=1`` reconstructs the noise-floor-clipped raw spectrum, i.e.
      smoothing is done on the spectrum in dB including noise
    - ``fill<1``: steeper edges, stronger suppression of isolated noise spikes
    - ``fill>>1``: converges to linear smoothing

    Boundaries are periodic (``mode='wrap'``), i.e. correct for folded spectra.

    Parameters
    ----------
    Z : xr.DataArray
        spectral reflectivity with dimension ``doppler`` (numpy or dask backed,
        ``doppler`` has to be a single chunk), noise subtracted, gaps set to 0
    noise : xr.DataArray
        noise level per bin, without the ``doppler`` dimension
    window : int, optional
        kernel length in bins, odd
    kernel : str, optional
        ``'box'``, ``'triangle'``, ``'gauss'`` or ``'savgol'``
    domain : str, optional
        ``'log'`` (default) or ``'lin'``
    fill : float, optional
        fraction of the noise floor used to fill the gaps (only ``'log'``)
    polyorder : int, optional
        polynomial order for ``kernel='savgol'``
    sigma : float, optional
        standard deviation in bins for ``kernel='gauss'``, default ``window/4``

    Returns
    -------
    xr.DataArray
        smoothed Z, values < 1e-8 set to 0
    """
    w = h.doppler_kernel(window, kernel, polyorder, sigma)
    return xr.apply_ufunc(
        h.smooth_doppler_np, Z, noise,
        input_core_dims=[['doppler'], []],
        output_core_dims=[['doppler']],
        kwargs=dict(weights=w, domain=domain, fill=fill),
        dask='parallelized',
        output_dtypes=[Z.dtype],
        keep_attrs=True,
    )
    


def load_rpgbinary(filename):
    """load the rpg LV0 files into a xr.datatree structure
    
    .. TODO::
        maybe track the chirp boundaries also in output file
    """

    header, data = read_rpg(filename)
    offset = (datetime.datetime(2001,1,1) - datetime.datetime(1970, 1, 1)).total_seconds()
    ts = offset*1e3 + data['Time']*1e3 + data['MSec']
    
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

        # Comparison with the LV1 files indicates, that the reflectivity might
        # be a factor of 2 (3dB) too high
        Z /= 2
        Zcx /= 2
        noise_combined /= 2

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
                time=('time', ts.astype('datetime64[ms]')),
                range=('range', rg[rg_chirp_map == sel_chirp]),
                doppler=('doppler', velocity_vector[sel_chirp,specslice]),
                ),
            #attrs=dict(),
        )
        datatree[f"chirp{sel_chirp+1}"] = xr.DataTree(ds)

    return datatree


def load_znc(filename, global_attrs=False):
    """load the Metek MIRA znc spectra into a xr.Dataset

    Parameters
    ----------
    filename : str or Path
        znc file
    global_attrs : bool, optional
        If True, copy the global attributes of the znc file and the radar
        parameters ``prf`` [Hz], ``nfft`` and ``nave`` into ``ds_input.attrs``.
    """


    # small time chunks: one chunk of (time, range, doppler) float32 is
    # ~1 MB per time step, and each dask thread holds several temporaries
    ds = xr.open_dataset(filename, chunks={'time': 100, 'range': -1, 'doppler': -1})
    #ds = xr.open_dataset(filename)
    print(ds.chunks)
    attrs = {}
    if global_attrs:
        attrs = dict(ds.attrs)
        attrs.update({v: ds[v].values.item() for v in ['prf', 'nfft', 'nave']})
    ds['time'] = (ds['time']*1e6 + ds['microsec']).astype('datetime64[us]')
    print(f"ds {ds.nbytes / 2**20:.2f} MiB")

    Z = ds['SPCco'] / ds['npw1'] * ds['RadarConst'] * (ds['range'] / 5e3)**2 * ds['SNRCorFaCo']
    Z.attrs['long_name'] = 'Spectral reflectivity'
    
    nfft = ds['doppler'].values.shape[0]
    #noise = ds['HSDco'] * ds['RadarConst'] * (ds['range'] / 5e3)**2 / nfft 
    # https://github.com/OPTIMICe-team/tripex_pol/blob/master/cheops_tripex_pol/ghostEchoFiltering_HPC/kaLib.py 
    noise = ds['HSDco'] / ds['npw1'] * ds['SNRCorFaCo'] * ds['RadarConst'] * (ds['range'] / 5e3)**2 
    noise_cx = ds['HSDcx'] / ds['npw2'] * ds['SNRCorFaCx'] * ds['RadarConst'] * (ds['range'] / 5e3)**2 
    
    Zcx = ds['SPCcx'] / ds['npw2'] * ds['RadarConst'] * (ds['range'] / 5e3)**2 * ds['SNRCorFaCx']
    
    no_roll = int(ds['doppler'].shape[0]/2)
    print('no_roll', no_roll)
    Z = Z.roll(doppler=no_roll, roll_coords=True).isel(doppler=slice(None, None, -1))
    Zcx = Zcx.roll(doppler=no_roll, roll_coords=True).isel(doppler=slice(None, None, -1))
    Z['doppler'] = Z['doppler']*-1
    Zcx['doppler'] = Zcx['doppler']*-1

    print(f"ds after chunk {ds.nbytes / 2**20:.2f} MiB")
    ds.close()
    del ds
    gc.collect()

    Z = Z - noise
    # equivalent to Z+noise < 0.5*noise
    Z = xr.where(Z < 1e-8, 0, Z)
    Zcx = Zcx - noise_cx
    Zcx = xr.where(Zcx < 1e-8, 0, Zcx)
    
    ds_input = xr.Dataset(
        data_vars={
            "Z": Z,
            "Zcx": Zcx,
            "noise": noise,
            "noise_cx": noise_cx,
        },
        attrs=attrs,
    )
    #print(ds_input['time'].shape)
    print(f"ds input {ds_input.nbytes / 2**20:.2f} MiB")

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


def ds_to_tree(ds_input: xr.Dataset, params: dict, meta: dict, time_block: int = 100):
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
    time_block : int, optional
        Number of time steps processed at once (default 100). Limits the
        size of the temporary stacked input array. If ``ds_input`` is lazy
        (dask-backed, e.g. from ``load_znc(filename)``), only one block is
        loaded into memory at a time. ``None`` disables blocking.

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

    var_peak = 'Z'
    # only stack the variables that are actually used, i.e. not the
    # (broadcasted) noise fields or the noise_mask
    inputvars = list(dict.fromkeys([var_peak, *meta.keys()]))
    vel_step = (ds_input['doppler'][1] - ds_input['doppler'][0]).values
    var_names = generate_tree.output_variable_names(meta)

    def apply_block(ds_block):
        # for lazy (dask) input this reads/computes only the current block,
        # so the full file never has to be in memory
        ds_block = ds_block[inputvars + ['noise_mask']].load()
        ds_block_array = ds_block[inputvars].to_array(dim="inputvar")
        return xr.apply_ufunc(
            generate_tree.ufunc_wrapper,
            ds_block_array.doppler,
            ds_block_array.inputvar,
            ds_block_array,
            ds_block['noise_mask'],
            kwargs={
                'vel_step':vel_step,
                'params': params,
                'var_peak': var_peak,
                'meta': meta },
            input_core_dims=[['doppler'], ['inputvar'], ['inputvar', 'doppler'], ['doppler']],
            output_core_dims=[['var', 'node']],
            output_dtypes=[np.float32],
            vectorize=True,
        )

    generate_tree.fix_peaks_unique.count = 0

    # process in blocks along time, so that the stacked copy of the input
    # only exists for one block at a time
    if time_block and 'time' in ds_input.dims and ds_input.sizes['time'] > time_block:
        n = ds_input.sizes['time']
        ds1 = xr.concat(
            [apply_block(ds_input.isel(time=slice(i, i + time_block)))
             for i in range(0, n, time_block)],
            dim='time')
    else:
        ds1 = apply_block(ds_input)

    if generate_tree.fix_peaks_unique.count:
        log.warning(
            "temporary fix for the prominence of equally high peaks applied "
            "to %d of %d spectra",
            generate_tree.fix_peaks_unique.count,
            int(np.prod([s for d, s in ds1.sizes.items() if d not in ('var', 'node')])))

    ds1.coords['var'] = var_names
    ds_rect = ds1.to_dataset(dim='var')
    ds_rect['no_nodes'] = (ds_rect['bounds_left'] != -999).sum(dim='node').astype('int32')

    for v in ['bounds_left', 'bounds_right', 'id_parent']:
        ds_rect[v] = ds_rect[v].astype('int32')
    node_ids = ds_rect.coords['node'].values.astype('int32')
    ds_rect['id_child_left'] = ('node', 2*node_ids + 1)
    ds_rect['id_child_right'] = ('node', 2*node_ids + 2)
    # node i has children if any node of the same spectrum has id_parent == i
    # (vectorized over all spectra instead of np.vectorize(h.has_children))
    id_parent = ds_rect['id_parent']
    node_ax = id_parent.get_axis_num('node')
    p = np.moveaxis(id_parent.values, node_ax, -1)
    has_ch = (p[..., np.newaxis, :] == np.arange(p.shape[-1])[:, np.newaxis]).any(axis=-1)
    ds_rect['has_children'] = id_parent.copy(data=np.moveaxis(has_ch, -1, node_ax))
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
    
