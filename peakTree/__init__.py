#! /usr/bin/env python3
# coding=utf-8
""""""
"""
Author: radenz@tropos.de
"""

import matplotlib
matplotlib.use('Agg')

import datetime
import logging
import ast
import subprocess
import re
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from . import helpers as h
from . import print_tree
from . import generate_tree
import toml
import scipy
from loess.loess_1d import loess_1d

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

from operator import itemgetter
from numba import jit

import scipy.signal
#import peakTree.fast_funcs as fast_funcs

from peakTree._meta import __version__, __author__

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
    delta_v = vel[~vel_mask][2] - vel[~vel_mask][1]
    
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


def saveVar(dataset, varData, dtype=np.float32):
    """save an item to the dataset with data in a dict

    Args:
        dataset (:obj:netCDF4.Dataset): netcdf4 Dataset to add to
        dtype: datatype of the array
        varData (dict): data to add, for example:

    ================== ======================================================
     Key                Description
    ================== ======================================================
    ``var_name``        name of the variable
    ``dimension``       ``('time', 'height')``
    ``arr``             data
    ``long_name``       descriptive long name
    **optional**
    ``comment``         description as a sentence
    ``units``           string with units
    ``units_html``      units in the html formatting
    ``missing_value``   define missing value
    ``plot_range``      tuple of range to plot
    ``plot_scale``      "linear" or "log" 
    ``axis``            
    ================== ======================================================
    """
    item = dataset.createVariable(varData['var_name'], dtype, 
            varData['dimension'], zlib=True, fill_value=varData['missing_value'])
    item[:] = np.ma.masked_less(varData['arr'], -990.)
    item.long_name = varData['long_name']
    if 'comment' in varData.keys():
        item.comment = varData['comment']
    if 'units' in varData.keys():
        item.units = varData['units']
    if 'units_html' in varData.keys():
        item.units_html = varData['units_html']
    if 'missing_value' in varData.keys():
        item.missing_value = varData['missing_value']
    if 'plot_range' in varData.keys():
        item.plot_range = varData['plot_range']
    if 'plot_scale' in varData.keys():
        item.plot_scale = varData['plot_scale']
    if 'axis' in varData.keys():
        item.axis = varData['axis']


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


def time_index(timestamps, sel_ts):
    """get the index of a timestamp in the list
    
    Args:
        timestamps: array
        sel_ts: timestamp whose index is required
    """
    return np.where(timestamps == min(timestamps, key=lambda t: abs(sel_ts - t)))[0][0]


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
    print(ts_range[0], ts_range[1])
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

    print(is_left[0], is_right[0])
    print(array[is_left[0]], array[0], array[is_right[0]])

    return zero_index + is_left, zero_index + is_right


def _roll_velocity(vel, vel_step, roll_vel, list_of_vars):
    """roll the spectrum, i.e., glue the rightmost x m/s to the left """
    #print('bin_roll_velocity ', roll_vel/vel_step)
    bin_roll_velocity = (roll_vel/vel_step).astype(int)
    velocity = np.concatenate((
        np.linspace(vel[0] - bin_roll_velocity * vel_step, 
                    vel[0] - vel_step, 
                    num=bin_roll_velocity), 
                    vel[:-bin_roll_velocity]))

    out_vars = []
    for var in list_of_vars:
        out_vars.append(
            np.concatenate((var[-bin_roll_velocity:], 
                            var[:-bin_roll_velocity]))
        )
    #specZ = np.concatenate((specZ[-bin_roll_velocity:], 
    #                        specZ[:-bin_roll_velocity]))
    #also specLDR, specZcx, specZ_mask, specZcx_mask, specLDR_mask
    return velocity, out_vars

class peakTreeBuffer():
    """trees for a time-height chunk

    Args:
        config_file (string, optional): path to the instrument config file (.toml)
        system (string, optional): specify the system/campaign

    The attribute setting may contain
    
    ==================== ========================================================
     Key                  Description
    ==================== ========================================================
    ``decoupling``        decoupling of the crosschannel
    ``smooth``            flag if smoothing should be applied
    ``grid_time``         time in seconds to average the spectra
    ``max_no_nodes``      number of nodes to save
    ``thres_factor_co``   factor between noise_lvl and noise thres in co channel
    ``thres_factor_cx``   factor between noise_lvl and noise thres in cross ch.
    ``station_altitude``  height of station above msl
    ==================== ========================================================
    
    """
    def __init__(self, config_file='instrument_config.toml', system="Lacros"):
        self.system = system

        with open(config_file) as cf:
            config = toml.loads(cf.read())

        if system in config:
            self.settings = config[system]["settings"]
            self.location = config[system]["location"]
            self.shortname = config[system]["shortname"]
        else:
            raise ValueError('no system defined')
        self.spectra_in_ram = False
        self.loader = config[system]['loader']

        self.peak_finding_params = config[system]['settings']['peak_finding_params']
        
        self.git_hash = get_git_hash()
        self.inputinfo = {}

        self.preload_grid = False


    def load(self, filename, load_to_ram=False):
        """convenience fuction for  loader call
        reads the self.loader (as specified in the instrument_config)
        """

        if self.loader == 'kazr_new':
            self.load_newkazr_file(filename, load_to_ram=load_to_ram)
        elif self.loader == 'kazr_legacy':
            self.load_kazr_file(filename, load_to_ram=load_to_ram)
        elif self.loader == 'rpg':
            self.load_rpgbinary_spec(filename, load_to_ram=load_to_ram)
        elif self.loader == 'rpgpy':
            self.load_rpgpy_spec(filename, load_to_ram=load_to_ram)
        else:
            self.load_spec_file(filename, load_to_ram=load_to_ram)

    def load_spec_file(self, filename, load_to_ram=False):
        """load spectra from raw file
        
        Args:
            filename: specify file
        """
        self.type = 'spec'

        self.f = netCDF4.Dataset(filename, 'r')
        print('keys ', self.f.variables.keys())

        self.timestamps = self.f.variables['time'][:]
        print('time ', self.timestamps[:10])
        self.delta_ts = np.mean(np.diff(self.timestamps)) if self.timestamps.shape[0] > 1 else 2.0
        self.range = self.f.variables['range'][:]
        print('range ', self.range[:10])
        self.velocity = self.f.variables['velocity'][:]
        print('velocity ', self.velocity[:10])
        print('Z chunking ', self.f.variables['Z'].chunking())

        self.begin_dt = h.ts_to_dt(self.timestamps[0])

        if load_to_ram == True:
            self.spectra_in_ram = True
            self.Z = self.f.variables['Z'][:].filled()
            self.LDR = self.f.variables['LDR'][:].filled()
            self.SNRco = self.f.variables['SNRco'][:].filled()



    def load_peakTree_file(self, filename):
        """load preprocessed peakTree file
     
        Args:
            filename: specify file
        """
        self.type = 'peakTree'
        self.f = netCDF4.Dataset(filename, 'r')
        log.info('loaded file {}'.format(filename))
        log.info('keys {}'.format(self.f.variables.keys()))
        self.timestamps = self.f.variables['timestamp'][:]
        self.delta_ts = np.mean(np.diff(self.timestamps)) if self.timestamps.shape[0] > 1 else 2.0
        self.range = self.f.variables['range'][:]
        self.no_nodes = self.f.variables['no_nodes'][:]


    def load_kazr_file(self, filename, load_to_ram=False):
        """load a kazr file
 
        Args:
            filename: specify file
        """ 
        self.type = 'kazr' 
        self.f = netCDF4.Dataset(filename, 'r') 
        print('loaded file ', filename) 
        print('keys ', self.f.variables.keys()) 
        #self.timestamps = self.f.variables['timestamp'][:] 
        time_offset = self.f.variables['time_offset']
        self.timestamps = self.f.variables['base_time'][:] + time_offset
        self.delta_ts = np.mean(np.diff(self.timestamps)) if self.timestamps.shape[0] > 1 else 2.0 
        self.range = self.f.variables['range'][:] 
        #self.velocity = self.f.variables['velocity'][:] 
        self.velocity = self.f.variables['velocity_bins'][:].astype(np.float64) 
        if isinstance(self.velocity, np.ma.MaskedArray):
            self.velocity = self.velocity.data
        assert not isinstance(self.velocity, np.ma.MaskedArray), \
            "velocity array shall not be np.ma.MaskedArray"

        # :cal_constant = "-24.308830 (dB)" ;
        # :cal_constant = "-12.8997 dB"
        self.cal_constant = float(re.findall("[+-]\d+.\d+", self.f.cal_constant)[0])
        self.cal_constant_lin = h.z2lin(self.cal_constant)
 
        self.begin_dt = h.ts_to_dt(self.timestamps[0])
        if load_to_ram == True:
            self.spectra_in_ram = True
            #self.Z = self.f.variables['spectra'][:]
            self.indices = self.f.variables['locator_mask'][:]
            self.spectra = self.f.variables['spectra'][:]


    def load_newkazr_file(self, filename, load_to_ram=False): 
        """load a kazr file
 
        Args:
            filename: specify file
        """ 
        self.type = 'kazr_new' 
        self.f = netCDF4.Dataset(filename, 'r') 
        print('loaded file ', filename) 
        print('keys ', self.f.variables.keys()) 
        #self.timestamps = self.f.variables['timestamp'][:] 
        time_offset = self.f.variables['time_offset']
        self.timestamps = self.f.variables['base_time'][:] + time_offset
        self.delta_ts = np.mean(np.diff(self.timestamps)) if self.timestamps.shape[0] > 1 else 2.0 
        self.range = self.f.variables['range'][:] 
        #self.velocity = self.f.variables['velocity'][:] 
        self.nyquist = self.f.variables['nyquist_velocity'][:] 
        assert np.all(self.nyquist[0] == self.nyquist)
        self.no_fft = self.f.dimensions["spectrum_n_samples"].size
        vel_res = 2 * self.nyquist[0] / float(self.no_fft)
        self.velocity = np.linspace(-self.nyquist[0] + (0.5 * vel_res),
                                    +self.nyquist[0] - (0.5 * vel_res),
                                    self.no_fft)
 
        self.begin_dt = h.ts_to_dt(self.timestamps[0])

        self.cal_constant = self.f.variables['r_calib_radar_constant_h'][:]
        assert self.cal_constant.shape[0] == 1
        self.cal_constant_lin = h.z2lin(self.cal_constant)

        if load_to_ram == True:
            self.spectra_in_ram = True
            #self.Z = self.f.variables['spectra'][:]
            self.indices = self.f.variables['spectrum_index'][:]
            self.spectra = self.f.variables['radar_power_spectrum_of_copolar_h'][:]


    def load_rpgbinary_spec(self, filename, load_to_ram=False):
        """load the rpg binary (.LV0) file directly; requires rpgpy


        Args:
            filename: path to file
            load_to_ram (optional False)


        2022-07-25: reimplemented the polarimetry based on sldr_bulk_revised-seeding_case.ipynb
                    and the communication with A. Myagkov

        developed with rpgpy version 0.10.2
        """
        from rpgpy import read_rpg, version

        self.type = 'rpgpy'
        header, data = read_rpg(filename)
        log.info(f'loaded file {filename} with rpgpy {version.__version__}')

        self.inputinfo = {
            'rpgpy': version.__version__, 'sw': header['SWVersion'],
        }
        log.debug(f'Header: {header.keys()}')
        log.debug(f'Data  : {data.keys()}')
        log.debug(f"no averaged spectra: {header['ChirpReps']/header['SpecN']}")
        Tc = 2.99e8/(4*header['MaxVel']*header['Freq']*1e9)
        log.debug(f"Tc [from Nyquist]: {Tc}")
        log.debug(f"Sample dur: {Tc*header['ChirpReps']} {np.sum(Tc*header['ChirpReps'])} {header['SampDur']}")
        log.debug(f"SeqIntTime: {header['SeqIntTime']}")


        offset = (datetime.datetime(2001,1,1) - datetime.datetime(1970, 1, 1)).total_seconds()
        self.timestamps = offset + data['Time'] + data['MSec']*1e-3
        self.delta_ts = np.mean(np.diff(self.timestamps)) if self.timestamps.shape[0] > 1 else 2.0
        self.range = header['RAlts']
        self.velocity = header['velocity_vectors'].T
        log.debug(f'velocity shape {self.velocity.shape}')

        self.chirp_start_indices = header['RngOffs']
        self.no_chirps = self.chirp_start_indices.shape[0]
        log.debug(f'chirp_start_indices {self.chirp_start_indices}')
        bins_per_chirp = np.diff(np.hstack((self.chirp_start_indices, self.range.shape[0])))
        log.debug(f'range bins per chirp {bins_per_chirp} {bins_per_chirp.shape}')
        # previously named n_samples_in_chirp
        self.doppFFT = header['SpecN'][:]
        self.rangeFFT = header['ChirpFFTSize'][:]
        self.no_avg = header['ChirpReps'][:]
        self.no_avg_subs = (2 * self.no_avg/self.doppFFT - 1) * (2 * self.rangeFFT -1)
        print((self.timestamps.shape[0], self.range.shape[0]))
        self.no_avg_subs_3d = np.broadcast_to(
            np.repeat(self.no_avg_subs, bins_per_chirp)[np.newaxis, :, np.newaxis], 
            (self.timestamps.shape[0], self.range.shape[0], np.max(self.doppFFT)))
        self.no_avg_subs_2d = np.broadcast_to(
            np.repeat(self.no_avg_subs, bins_per_chirp)[np.newaxis, :], 
            (self.timestamps.shape[0], self.range.shape[0]))

        self.Q = header['NoiseFilt']

        self.range_chirp_mapping = np.repeat(np.arange(self.no_chirps), bins_per_chirp)
        self.begin_dt = h.ts_to_dt(self.timestamps[0])

        scaling = self.settings['tot_spec_scaling']
        log.warning(f"WARNING: Taking scaling factor from config file. It should be 1 for RPG software version > 5.40, "
                    f"and 0.5 for earlier software versions (around 2020). Only applicable for STSR radar. Configured "
                    f"are {scaling}")

        if load_to_ram == True:
            self.spectra_in_ram = True
            #print(type(data['TotSpec']), data['TotSpec'])
            self.spec_tot = data['TotSpec']
            self.spec_tot = scaling*self.spec_tot

            # TODO: Check distinction between STSR and SP radar: For STSR, VNoisePow is stored in the "TotNoisePow"
            #  variable (this is a problem in rpgpy with variable naming). VNoisePow and HNoisePow have to be added up
            #  to get the total noise. For SP, the total noise is stored in TotNoisePow and there is no variable named
            #  HNoisePow. [LDR mode radar??]
            #  If no compression is applied in the software, complete spectra are saved and the noise power is not at
            #  all stored in the files (no matter which polarimetric type of RPG radar) and needs to be computed here.
            #  TBD: Do we need to also apply the scaling factor to the noise?

            if self.settings['polarimetry'] == 'STSR':
                # possibly missing scaling factor here:
                if 'TotNoisePow' in data:
                    self.noise_v = data['TotNoisePow'] 
                else:
                    self.noise_v = h.estimate_noise_array(self.spec_tot)
                self.noise_v /= np.repeat(self.doppFFT, bins_per_chirp) 
                self.noise_h = data['HNoisePow']/np.repeat(self.doppFFT, bins_per_chirp)

                self.spec_h = data['HSpec']
                self.spec_cov_re = data['ReVHSpec']
                self.spec_cov_im = data['ImVHSpec']
                self.spec_v = 4 * self.spec_tot - self.spec_h - 2 * self.spec_cov_re

                self.noise_v_3d = np.repeat(self.noise_v[:,:,np.newaxis], np.max(self.doppFFT), axis=2)
                self.noise_h_3d = np.repeat(self.noise_h[:,:,np.newaxis], np.max(self.doppFFT), axis=2)
                self.noise_combined_3d = (self.noise_v_3d + self.noise_h_3d) / 2
                self.noise_combined = (self.noise_v + self.noise_h) / 2
                sv = self.spec_v.copy()
                sv += self.noise_v_3d
                sh = self.spec_h.copy()
                sh += self.noise_h_3d

                self.rhv_2d = np.abs(self.spec_cov_re + 1j * self.spec_cov_im) / np.sqrt(sv * sh)

                self.specZ_2d = (sv + sh)*(1+self.rhv_2d) / 2 - self.noise_combined_3d
                self.specZcx_2d = (sv + sh)*(1-self.rhv_2d) / 2 - self.noise_combined_3d

                self.specZ_2d_mask = (self.specZ_2d <= 1e-10)
                self.specZcx_2d_mask = (self.specZ_2d <= 1e-10) | (self.specZcx_2d <= 1e-10)

                self.noise_thres_2d = self.Q*self.noise_combined/np.sqrt(self.no_avg_subs_2d)

            elif self.settings['polarimetry'] == 'false':
                if 'TotNoisePow' in data:
                    self.noise_v = data['TotNoisePow'] 
                else:
                    self.noise_v = h.estimate_noise_array(self.spec_tot)
                self.specZ_2d = self.spec_tot
                self.specZ_2d_mask = (self.specZ_2d <= 1e-10)
                # here another option for polarimetry = 'LDR' needs to be added

        else:
            raise ValueError('load_to_ram = False not implemented yet')


    def load_rpgpy_spec(self, filename, load_to_ram=False):
        """ WIP implementation of the rpgpy spectra format

        See https://github.com/actris-cloudnet/rpgpy
        """

        #TODO include the polarization state (variable dual_polarization in rpgpy netcdf)
        log.warning(f"WARNING: not actively maintained use binary version instead (loader='rpg')")

        self.type = 'rpgpy'
        self.f = netCDF4.Dataset(filename, 'r')
        print('loaded file ', filename)
        print('keys ', self.f.variables.keys())
        #times = self.f.variables['decimal_time'][:]
        #seconds = times.astype(np.float)*3600.
        #self.timestamps = h.dt_to_ts(datetime.datetime(2014, 2, 21)) + seconds
        self.inputinfo = {
            'sw': self.f.variables['software_version'],
        }

        # convention here is unix time rpgpy provides since beginning of 2001 with additional milliseconds
        time = self.f.variables['time'][:]
        time_ms = self.f.variables['time_ms'][:]
        offset = (datetime.datetime(2001,1,1) - datetime.datetime(1970, 1, 1)).total_seconds()
        self.timestamps = offset + time + time_ms*1e-3
        self.delta_ts = np.mean(np.diff(self.timestamps)) if self.timestamps.shape[0] > 1 else 2.0 
        self.range = self.f.variables['range_layers'][:]
        self.velocity = self.f.variables['velocity_vectors'][:].T
        print('velocity shape', self.velocity.shape)
        
        self.chirp_start_indices = self.f.variables['chirp_start_indices'][:]
        self.n_samples_in_chirp = self.f.variables['n_samples_in_chirp'][:]
        self.no_chirps = self.chirp_start_indices.shape[0]
        print('chirp_start_indices', self.chirp_start_indices)
        bins_per_chirp = np.diff(np.hstack((self.chirp_start_indices, self.range.shape[0])))
        print('range bins per chirp', bins_per_chirp, bins_per_chirp.shape)

        self.range_chirp_mapping = np.repeat(np.arange(self.no_chirps), bins_per_chirp) 
        self.begin_dt = h.ts_to_dt(self.timestamps[0])

        scaling = self.settings['tot_spec_scaling']
        log.warning(f"WARNING: Taking scaling factor from config file. It should be 1 for RPG software version > 5.40, "
                    f"and 0.5 for earlier software versions (around 2020). Only applicable for STSR radar. Configured "
                    f"are {scaling}")

        if load_to_ram == True:
            self.spectra_in_ram = True
            self.doppler_spectrum, spec_mask = h.masked_to_plain(self.f.variables['doppler_spectrum'][:])
            self.doppler_spectrum = scaling*self.doppler_spectrum


            if self.settings['polarimetry'] == 'STSR':
                self.doppler_spectrum_h, spec_h_mask = h.masked_to_plain(self.f.variables['doppler_spectrum_h'][:])
                self.covariance_spectrum_re, cov_re_mask = h.masked_to_plain(self.f.variables['covariance_spectrum_re'][:])
                self.covariance_spectrum_im, cov_im_mask = h.masked_to_plain(self.f.variables['covariance_spectrum_im'][:])
                self.spectral_mask = (spec_mask | spec_h_mask | cov_re_mask | cov_im_mask)
            else:
                self.spectral_mask = spec_mask

            if 'integrated_noise' in self.f.variables:
                self.integrated_noise, _ = h.masked_to_plain(self.f.variables['integrated_noise'][:])
            else:
                self.integrated_noise = h.estimate_noise_array(self.doppler_spectrum)*np.repeat(self.n_samples_in_chirp, bins_per_chirp)
            self.doppler_spectrum[self.spectral_mask] = 0
            self.integ_noise_per_bin = (self.integrated_noise/np.repeat(self.n_samples_in_chirp, bins_per_chirp))

            if self.settings['polarimetry'] == 'STSR':
                self.doppler_spectrum_h[self.spectral_mask] = 0
                self.covariance_spectrum_re[self.spectral_mask] = np.nan
                self.covariance_spectrum_im[self.spectral_mask] = np.nan
                self.integrated_noise_h, _ = h.masked_to_plain(self.f.variables['integrated_noise_h'][:])


                self.doppler_spectrum_v = 4 * self.doppler_spectrum - self.doppler_spectrum_h - 2 * self.covariance_spectrum_re
                noise_v = self.integrated_noise / 2.

                print('shapes, noise per bin', self.integrated_noise_h.shape, np.repeat(self.n_samples_in_chirp, bins_per_chirp).shape)
                self.noise_h_per_bin = (self.integrated_noise_h/np.repeat(self.n_samples_in_chirp, bins_per_chirp))
                print(self.noise_h_per_bin.shape)
                #self.noise_h_per_bin = np.repeat(noise_h_per_bin[:,:,np.newaxis], self.velocity.shape[0], axis=2)
                self.noise_v_per_bin = (noise_v/np.repeat(self.n_samples_in_chirp, bins_per_chirp)) 
            #self.noise_v_per_bin = np.repeat(noise_v_per_bin[:,:,np.newaxis], self.velocity.shape[0], axis=2)

            #rhv = np.abs(np.complex(cov_re, cov_im)/np.sqrt(()*(spec_chunk_h + spec_noise_h)))
            #print('shapes ', self.covariance_spectrum_re.shape, self.covariance_spectrum_im.shape, self.doppler_spectrum_v.shape, noise_v_per_bin.shape, 
            #    self.doppler_spectrum_h.shape, noise_h_per_bin.shape)
            #self.rhv = np.abs(self.covariance_spectrum_re + 1j * self.covariance_spectrum_im) / np.sqrt( 
            #    (self.doppler_spectrum_v + self.noise_v_per_bin[:,:,np.newaxis]) * (self.doppler_spectrum_h + self.noise_h_per_bin[:,:,np.newaxis]) )



    def preload_averaging_grid(self):
        """precalculate the averaging boundaries
        
        """
        self.preload_grid = True

        time_grid = {}        
        if self.type == 'rpgpy':
            for ind_chirp in range(self.no_chirps):
                # t_avg = 0 gives wrong boundaries
                t_avg = max(0.0001, self.peak_finding_params[f"chirp{ind_chirp}"]['t_avg'])
                time_grid[ind_chirp] = get_averaging_boundaries(
                    self.timestamps, t_avg)
        else:
            t_avg = max(0.0001, self.peak_finding_params['t_avg'])
            time_grid[0] = get_averaging_boundaries(
                self.timestamps, t_avg)

    
        if self.type == 'rpgpy':
            left, right = [], []
            for ind_chirp in range(self.no_chirps):
                rg_avg = max(0.01, self.peak_finding_params[f"chirp{ind_chirp}"]['h_avg'])
                # bounds of the current chirp
                ir_min, ir_max = np.where(self.range_chirp_mapping == ind_chirp)[0][np.array([0, -1])]
                il, ir = get_averaging_boundaries(
                    self.range[ir_min:ir_max+1], rg_avg, zero_index=ir_min)
                left.append(il)
                right.append(ir)
            range_grid = [np.concatenate(left), np.concatenate(right)]
        else:
            ind_chirp = 0
            rg_avg = max(0.01, self.peak_finding_params['h_avg'])
            range_grid = get_averaging_boundaries(
                self.range, rg_avg)
            #temp_avg = self.peak_finding_params['t_avg']

        self.time_grid = time_grid
        self.range_grid = range_grid


    def get_it_interval(self, it, ir, sel_ts=None):
        """"""
        if self.type == 'rpgpy':
            ind_chirp = self.range_chirp_mapping[ir]
            temp_avg = self.peak_finding_params[f"chirp{ind_chirp}"]['t_avg']
        else:
            ind_chirp = 0
            temp_avg = self.peak_finding_params['t_avg']

        if self.preload_grid:
            it_b = self.time_grid[ind_chirp][0][it]
            it_e = self.time_grid[ind_chirp][1][it]
        else:
            if not sel_ts:
                sel_ts = self.timestamps[it]
            temp_avg = self.peak_finding_params['t_avg']
            it_b = time_index(self.timestamps, sel_ts-temp_avg/2.)
            it_e = time_index(self.timestamps, sel_ts+temp_avg/2.)+1

        return it_b, it_e

    def get_ir_interval(self, ir, sel_range=None):
        """"""
        if self.type == 'rpgpy':
            ind_chirp = self.range_chirp_mapping[ir]
            rg_avg = self.peak_finding_params[f"chirp{ind_chirp}"]['h_avg']
        else:
            ind_chirp = 0
            rg_avg = self.peak_finding_params['h_avg']

        if self.preload_grid:
            ir_b = self.range_grid[0][ir]
            ir_e = self.range_grid[1][ir]
        else:
            rg_avg = self.peak_finding_params['h_avg']
            if not sel_range:
                sel_range = self.range[ir]
            ir_b = np.where(self.range == min(self.range, key=lambda t: abs(sel_range - rg_avg/2. - t)))[0][0]
            ir_e = np.where(self.range == min(self.range, key=lambda t: abs(sel_range + rg_avg/2. - t)))[0][0]+1

        return ir_b, ir_e

    #@profile
    def get_tree_at(self, sel_ts, sel_range, temporal_average='fromparams', 
                    roll_velocity=False, peak_finding_params={}, 
                    tail_filter=False,
                    silent=False):
        """get a single tree
        either from the spectrum directly (prior call of load_spec_file())
        or from the pre-converted file (prior call of load_peakTree_file())

        Args:
            sel_ts (tuple or float): either value or (index, value)
            sel_range (tuple or float): either value or (index, value)
            temporal_average (optional): average over the interval 
                (tuple of bin in time dimension or 'fromparams' or False)
            roll_velocity (optional): shift the x rightmost bins to the
                left 
            peak_finding_params (optional):
            silent: verbose output

        'vel_smooth': convolution with given array 
        'prom_thres': prominence threshold in dB
        
        Returns:
            dictionary with all nodes and the parameters of each node
        """


        if type(sel_ts) is tuple and type(sel_range) is tuple:
            it, sel_ts = sel_ts
            ir, sel_range = sel_range
        else:
            it = time_index(self.timestamps, sel_ts)
            ir = np.where(self.range == min(self.range, key=lambda t: abs(sel_range - t)))[0][0]
        #it_b, it_e = it, min(it+1, self.timestamps.shape[0]-1)
        log.info('time {} {} {} height {} {}'.format(it, h.ts_to_dt(self.timestamps[it]), self.timestamps[it], ir, self.range[ir]))
        assert np.abs(sel_ts - self.timestamps[it]) < self.delta_ts, 'timestamps more than '+str(self.delta_ts)+'s apart'
        #assert np.abs(sel_range - self.range[ir]) < 10, 'ranges more than 10m apart'

        #assert self.preload_grid
        ir_min, ir_max = 0, self.range.shape[0]-1
        if self.type == 'rpgpy':
            ind_chirp = self.range_chirp_mapping[ir]
            peak_finding_params = self.peak_finding_params[f"chirp{ind_chirp}"]
            ir_min, ir_max = np.where(self.range_chirp_mapping == ind_chirp)[0][np.array([0, -1])]
            #print('ir_min, ir_max', ir_min, ir_max)
        peak_finding_params = (lambda d: d.update(peak_finding_params) or d)(self.peak_finding_params)
        if 'smooth_in_dB' not in peak_finding_params:
            peak_finding_params['smooth_in_dB'] = True
        log.debug(f'using peak_finding_params {peak_finding_params}')
        ir_b, ir_e = self.get_ir_interval(ir) 
        ir_slicer = slice(max(ir_b, ir_min), min(ir_e, ir_max))

        # decouple temporal average from grid 
        # 
        it_b, it_e = self.get_it_interval(it, ir) 
        it_slicer = slice(it_b, min(it_e, self.timestamps.shape[0]-1))
        it_slicer = slice(it_b, it_e)
        # be extremely careful with slicer and _e or _e+1
        slicer_sel_ts = self.timestamps[it_slicer]
        log.debug('timerange {} {} {} '.format(str(it_slicer), h.ts_to_dt(slicer_sel_ts[0]), h.ts_to_dt(slicer_sel_ts[-1]))) 
        assert slicer_sel_ts[-1] - slicer_sel_ts[0] < 20, 'found averaging range too large'

        #print('ir selected', ir_min, ir_b, ir, ir_e, ir_max, ir_slicer, 
        #      '      it ', it_b, it, it_e, slicer_sel_ts.tolist())
        #print(self.range[max(ir_b, ir_min)], self.range[ir], self.range[min(ir_e, ir_max)])

        if self.type == 'spec':
            decoupling = self.settings['decoupling']

            # why is ravel necessary here?
            # flatten seems to faster

            if self.spectra_in_ram:
                specZ_2d = self.Z[:,ir_slicer,it_slicer][:]
                no_averages = np.prod(specZ_2d.shape[:-1])
                specLDR_2d = self.LDR[:,ir_slicer,it_slicer][:]
                specZcx_2d = specZ_2d*specLDR_2d
                # np.average is slightly faster than .mean
                #specZcx = specZcx_2d.mean(axis=1)
                specZcx = np.average(specZcx_2d, axis=(1,2))
                #specZ = specZ_2d.mean(axis=1)
                specZ = np.average(specZ_2d, axis=(1,2))
            else:
                specZ = self.f.variables['Z'][:,ir_slicer,it_slicer][:].filled()
                no_averages = specZ.shape[1]
                specLDR = self.f.variables['LDR'][:,ir_slicer,it_slicer][:].filled()
                specZcx = specZ*specLDR
                specZcx = specZcx.mean(axis=(1,2))
                specZ = specZ.mean(axis=(1,2))

            specLDR = specZcx/specZ
            assert not isinstance(specZ, np.ma.core.MaskedArray), "Z not np.ndarray"
            assert not isinstance(specZcx, np.ma.core.MaskedArray), "Z not np.ndarray"
            assert not isinstance(specLDR, np.ma.core.MaskedArray), "LDR not np.ndarray"
            #print('specZ ', type(specZ), h.lin2z(specZ[120:-120]))
            #print('specZcx ', type(specZcx), h.lin2z(specZcx[120:-120]))

            # fill values can be both nan and 0
            specZ_mask = np.logical_or(~np.isfinite(specZ), specZ == 0)
            empty_spec = np.all(specZ_mask)
            if empty_spec:
                specZcx_mask = specZ_mask.copy()
                specLDR_mask = specZ_mask.copy()
            elif np.all(np.isnan(specZcx)):
                specZcx_mask = np.ones_like(specZcx).astype(bool)
                specLDR_mask = np.ones_like(specLDR).astype(bool)
            else:
                specZcx_mask = np.logical_or(~np.isfinite(specZ), ~np.isfinite(specZcx))
                specLDR_mask = np.logical_or(specZ == 0, ~np.isfinite(specLDR))
                # maybe omit one here?
            assert np.all(specZcx_mask == specLDR_mask), f'masks not equal {specZcx} {specLDR}'
            #specSNRco_mask = specSNRco == 0.
            # print('specZ', specZ.shape, specZ)
            # print('specLDR', specLDR.shape, specLDR)
            # print('specSNRco', specSNRco.shape, specSNRco)

            #specSNRco = np.ma.masked_equal(specSNRco, 0)
            #noise_thres = 1e-25 if empty_spec else np.min(specZ[~specZ_mask])*peak_finding_params['thres_factor_co']
            noise_thres_old = 1e-25 if empty_spec else np.min(specZ[~specZ_mask])*peak_finding_params['thres_factor_co']
            if not self.spectra_in_ram:
                noise_thres = noise_thres_old
            # alternate fit estimate for the problematic mira spectra 
            else:
                noise_thres = 1e-25 if empty_spec else np.nanmin(specZ_2d[specZ_2d > 0])*peak_finding_params['thres_factor_co']
            print(f'noise est {h.lin2z(noise_thres_old):.2f} -> {h.lin2z(noise_thres):.2f}')

            velocity = self.velocity.copy()
            vel_step = velocity[1] - velocity[0]
            if roll_velocity or ('roll_velocity' in peak_finding_params and peak_finding_params['roll_velocity']):
                if 'roll_velocity' in peak_finding_params and peak_finding_params['roll_velocity']:
                    roll_velocity = peak_finding_params['roll_velocity']
                log.info(f'>> roll velocity active {roll_velocity}')
                upck = _roll_velocity(velocity, vel_step, roll_velocity, 
                                      [specZ, specLDR, specZcx, specZ_mask, specZcx_mask, specLDR_mask])
                velocity = upck[0]
                specZ, specLDR, specZcx, specZ_mask, specZcx_mask, specLDR_mask = upck[1]

            window_length = h.round_odd(peak_finding_params['span']/vel_step)
            log.debug(f"window_length {window_length},  polyorder  {peak_finding_params['smooth_polyorder']}")

            specZ_raw = specZ.copy()

            # for noise separeated peaks 0 is a better fill value
            if type(tail_filter) == list:
                noise_tail = h.gauss_func_offset(velocity, *tail_filter)
                log.warning(f'tail filter applied {tail_filter}')
                specZ[specZ < h.z2lin(noise_tail + 4)] = np.nan
                tail_filter_applied = True
                #print('tail filter applied, ', tail_filter_applied, tail_filter)
            else:
                tail_filter_applied = False
        
            # --------------------------------------------------------------------
            # smoothing and cutting. the order can be defined in instrument_config
            if self.settings['smooth_cut_sequence'] == 'cs':
                specZ[specZ < noise_thres] = np.nan #noise_thres / 6. 
            if peak_finding_params['smooth_polyorder'] != 0:
                specZ = h.lin2z(specZ)
                if peak_finding_params['smooth_polyorder'] < 10:
                    specZ = scipy.signal.savgol_filter(specZ, window_length, 
                                polyorder=peak_finding_params['smooth_polyorder'], 
                                mode='nearest')
                elif 10 < peak_finding_params['smooth_polyorder'] < 20:
                    if indices:
                        _, specZ, _ = loess_1d(velocity, specZ, 
                                    degree=peak_finding_params['smooth_polyorder']-10, 
                                    npoints=window_length)
                elif 20 < peak_finding_params['smooth_polyorder'] < 30:
                    window = h.gauss_func(np.arange(11), 5, window_length)
                    specZ = np.convolve(specZ, window, mode='same')
                else:
                    raise ValueError(f"smooth_polyorder = {peak_finding_params['smooth_polyorder']} not defined")
                specZ = h.z2lin(specZ)
            gaps = (specZ <= 0.) | ~np.isfinite(specZ)
            # this collides with the tail filter
            #specZ[gaps] = specZ_raw[gaps]
            if self.settings['smooth_cut_sequence'] == 'sc':
                specZ[specZ < noise_thres] = np.nan #noise_thres / 6. 
            
            # TODO add for other versions
            #specZ_mask = (specZ_mask) | (specZ < noise_thres) | ~np.isfinite(specZ)
            specZ_mask = (specZ < noise_thres) | ~np.isfinite(specZ)

            peak_finding_params['vel_step'] = vel_step
            #specZ[specZ_mask] = 0

            spectrum = {'ts': self.timestamps[it], 'range': self.range[ir],
                        'noise_thres': noise_thres, 'no_temp_avg': no_averages}
            spectrum['specZ'] = specZ[:]
            spectrum['vel'] = velocity

            if tail_filter_applied:     
                spectrum['tail_filter'] = tail_filter

            # unsmoothed spectrum
            spectrum['specZ_raw'] = specZ_raw[:]
            
            spectrum['specZ_mask'] = specZ_mask[:]
            #spectrum['specSNRco'] = specSNRco[:]
            #spectrum['specSNRco_mask'] = specSNRco_mask[:]
            spectrum['specLDR'] = specLDR[:]
            spectrum['specLDR_mask'] = specLDR_mask[:]
            
            spectrum['specZcx'] = specZcx[:]
            # spectrum['specZcx_mask'] = np.logical_or(spectrum['specZ_mask'], spectrum['specLDR_mask'])
            if ('thres_factor_cx' in peak_finding_params 
                and peak_finding_params['thres_factor_cx']):
                noise_cx_thres = np.nanmin(spectrum['specZcx']) * peak_finding_params['thres_factor_cx']
            else:
                noise_cx_thres = noise_mean
            specZcx_mask = (specZcx_mask | (specZcx < noise_cx_thres))
            spectrum['noise_cx_thres'] = noise_cx_thres

            trust_ldr_mask = specZcx_mask | ~np.isfinite(spectrum['specZcx']) | specZ_mask
            spectrum['trust_ldr_mask'] = trust_ldr_mask
            spectrum['specLDRmasked'] = spectrum['specLDR'].copy()
            spectrum['specLDRmasked'][trust_ldr_mask] = np.nan

            spectrum['decoupling'] = decoupling
            spectrum['polarimetry'] = self.settings['polarimetry']

            #                                     deep copy of dict 
            #travtree = generate_tree.tree_from_spectrum({**spectrum}, peak_finding_params)
            travtree = generate_tree.tree_from_spectrum_peako(
                #{**spectrum}, peak_finding_params, gaps=gaps) 
                {**spectrum}, peak_finding_params, gaps=None) 
            #travtree = {}

            if (travtree 
                 and 'tail_filter' in peak_finding_params
                 and peak_finding_params['tail_filter'] is True):
                no_ind = (travtree[0]['bounds'][1]- travtree[0]['bounds'][0])
                #print('tail criterion? ', no_ind * vel_step, '>', 9 * travtree[0]['width']) 
            # an agressive tail filter would have to act here?
            if (travtree 
                 and 'tail_filter' in peak_finding_params
                 and peak_finding_params['tail_filter'] is True
                 and h.lin2z(travtree[0]['z']) > -3 
                 #and travtree[0]['width'] < 0.22 and not tail_filter_applied):
                 #and travtree[0]['width'] < 0.31 
                 and no_ind * vel_step > 9 * travtree[0]['width'] 
                 and not tail_filter_applied):

                # strategy 1 fit here
                # alternative parametrize more strongly
                log.warning(f'Tails might occur here {sel_ts} {sel_range}')
                ind_vel_node0 = np.searchsorted(spectrum['vel'], travtree[0]['v'])
                no_ind *= 0.40
                fit_spec = h.lin2z(specZ_raw.copy())
                #fit_spec[ind_vel_node0-int(no_ind/2):ind_vel_node0+int(no_ind/2)] = np.nan
                fit_spec[fit_spec > h.lin2z(noise_thres) + 13] = np.nan
                print('noise thres ', h.lin2z(noise_thres), h.lin2z(noise_thres) + 15)
                fit_mask = np.isfinite(fit_spec)
                try:
                    popt, _ = h.gauss_fit(spectrum['vel'][fit_mask], fit_spec[fit_mask])
                    successful_fit = True
                except:
                    successful_fit = False
                    log.warning('fit failed')
                    #input()

                if successful_fit:
                    travtree, spectrum = self.get_tree_at(
                        sel_ts, sel_range, temporal_average=temporal_average, 
                        roll_velocity=roll_velocity, peak_finding_params=peak_finding_params, 
                        tail_filter=popt.tolist(), silent=silent)
            
            return travtree, spectrum

        elif self.type == 'peakTree':
            settings_file = ast.literal_eval(self.f.settings)
            self.settings['max_no_nodes'] = settings_file['max_no_nodes']
            log.info('load tree from peakTree; no_nodes {}'.format(self.no_nodes[it,ir])) if not silent else None
            travtree = {}            
            log.debug('peakTree parent {}'.format(self.f.variables['parent'][it,ir,:])) if not silent else None

            avail_nodes = np.argwhere(~self.f.variables['parent'][it,ir,:].mask).ravel()
            for k in avail_nodes.tolist():
                #print('k', k)
                #(['timestamp', 'range', 'Z', 'v', 'width', 'LDR', 'skew', 'minv', 'maxv', 'threshold', 'parent', 'no_nodes']
                node = {'parent_id': int(np.asscalar(self.f.variables['parent'][it,ir,k])), 
                        'thres': h.z2lin(np.asscalar(self.f.variables['threshold'][it,ir,k])), 
                        'width': np.asscalar(self.f.variables['width'][it,ir,k]), 
                        #'bounds': self.f.variables[''][it,ir], 
                        'z': h.z2lin(np.asscalar(self.f.variables['Z'][it,ir,k])), 
                        'bounds': (np.asscalar(self.f.variables['bound_l'][it,ir,k]), np.asscalar(self.f.variables['bound_r'][it,ir,k])),
                        #'coords': [0], 
                        'skew': np.asscalar(self.f.variables['skew'][it,ir,k]),
                        'prominence': h.z2lin(np.asscalar(self.f.variables['prominence'][it,ir,k])),
                        'v': np.asscalar(self.f.variables['v'][it,ir,k])}
                if 'LDR' in self.f.variables.keys():
                    node['ldr'] = h.z2lin(np.asscalar(self.f.variables['LDR'][it,ir,k]))
                    node['ldrmax'] =  h.z2lin(np.asscalar(self.f.variables['ldrmax'][it,ir,k]))
                else:
                    node['ldr'], node['ldrmax'] = -99, -99
                if node['parent_id'] != -999:
                    if k == 0:
                        node['coords'] = [0]
                    else:
                        coords = travtree[node['parent_id']]['coords']
                        if k%2 == 0:
                            node['coords'] = coords + [1]
                        else:
                            node['coords'] = coords + [0]
                        #siblings = list(filter(lambda d: d['coords'][:-1] == coords, travtree.values()))
                        # #print('parent_id', node['parent_id'], 'siblings ', siblings)
                        #node['coords'] = coords + [len(siblings)]
                    travtree[k] = node

            return travtree, None

        elif self.type == 'kazr': 
             
            if not temporal_average:
                if self.spectra_in_ram:
                    index = self.indices[it,ir]
                    specZ = h.z2lin(self.spectra[index,:])
                else:
                    index = self.f.variables['locator_mask'][it,ir]
                    specZ = h.z2lin(self.f.variables['spectra'][index,:])
                    #specZ = self.f.variables['spectra'][it,ir,:] 
                no_averages = 1 
                specZ = specZ * self.cal_constant_lin * self.range[ir]**2
                specZ_mask = specZ == 0. 
            else:
                if self.spectra_in_ram:
                    indices = self.indices[it_b:it_e+1,ir].tolist()
                    #print(indices)
                    indices = [i for i in indices if i is not None]
                    if indices:
                        specZ = h.z2lin(self.spectra[indices,:])
                    else:
                        #empty spectrum
                        specZ = np.full((2, self.velocity.shape[0]), h.z2lin(-70))
                else: 
                    indices = self.f.variables['locator_mask'][it_b:it_e+1,ir].tolist()
                    indices = [i for i in indices if i is not None]
                    if indices:
                        specZ = h.z2lin(self.f.variables['spectra'][indices,:])
                    else:
                        #empty spectrum
                        specZ = np.full((2, self.velocity.shape[0]), h.z2lin(-70))

                #if isinstance(specZ, np.ma.MaskedArray):
                #    specZ = specZ.data
                #assert not (isinstance(specZ, np.ma.MaskedArray)\
                #        or isinstance(specZ, np.ma.core.MaskedArray)), \
                #    "specZ array shall not be np.ma.MaskedArray"
                no_averages = specZ.shape[0] 
                specZ = np.average(specZ, axis=0)
                specZ = specZ * self.cal_constant_lin * self.range[ir]**2
                specZ_mask = np.logical_or(~np.isfinite(specZ), specZ == 0)  
 
            noise = h.estimate_noise(specZ, no_averages) 
            #print("nose_thres {:5.3f} noise_mean {:5.3f} no noise bins {}".format( 
            #    h.lin2z(noise['noise_sep']), h.lin2z(noise['noise_mean']), 
            #    noise['no_noise_bins'])) 
            noise_mean = noise['noise_mean'] 
            #noise_thres = noise['noise_sep'] 
            noise_thres = noise['noise_mean']*peak_finding_params['thres_factor_co']
 
            velocity = self.velocity
            if roll_velocity or ('roll_velocity' in self.settings and self.settings['roll_velocity']):
                if 'roll_velocity' in self.settings and self.settings['roll_velocity']:
                    roll_velocity = self.settings['roll_velocity']
                vel_step = self.velocity[1] - self.velocity[0]
                velocity = np.concatenate((
                    np.linspace(self.velocity[0] - roll_velocity * vel_step, 
                                self.velocity[0] - vel_step, 
                                num=roll_velocity), 
                    self.velocity[:-roll_velocity]))

                specZ = np.concatenate((specZ[-roll_velocity:], 
                                        specZ[:-roll_velocity]))
                specZ_mask = np.concatenate((specZ_mask[-roll_velocity:], 
                                             specZ_mask[:-roll_velocity]))


            if 'span' in peak_finding_params:
                window_length = h.round_odd(peak_finding_params['span']/(self.velocity[1]-self.velocity[0]))
                log.debug(f"window_length {window_length},  polyorder  {peak_finding_params['smooth_polyorder']}")
                specZ = scipy.signal.savgol_filter(specZ, window_length, 
                        polyorder=peak_finding_params['smooth_polyorder'], mode='nearest')
            else:
                if 'vel_smooth' in peak_finding_params and type(peak_finding_params['vel_smooth']) == list:
                    print('vel_smooth based on list')
                    convol_window = peak_finding_params['vel_smooth']
                    print('convol_window ', convol_window)
                    specZ = np.convolve(specZ, convol_window, mode='same')
                elif 'vel_smooth' in peak_finding_params:
                    convol_window = np.array([0.5,1,0.5])/2.0
                    print('convol_window ', convol_window)
                    specZ = np.convolve(specZ, convol_window, mode='same')
                else:
                    print("! no smoothing applied")
             
 
            specSNRco = specZ/noise_mean 
            specSNRco_mask = specZ.copy() 
 
 
            spectrum = { 
                'ts': self.timestamps[it], 'range': self.range[ir],  
                'vel': velocity, 
                'polarimetry': self.settings['polarimetry'],
                'specZ': specZ, 'noise_thres': noise_thres, 
                'specZ_mask': specZ_mask, 
                'no_temp_avg': no_averages, 
                'specSNRco': specSNRco, 
                'specSNRco_mask': specSNRco_mask 
            } 

            travtree = generate_tree.tree_from_spectrum({**spectrum}, peak_finding_params) 
            return travtree, spectrum 

        elif self.type == 'kazr_new': 

            #self.indices = self.f.variables['spectrum_index'][:]
            #self.spectra = self.f.variables['radar_power_spectrum_of_copolar_h'][:]
             
            print('slicer ', it_slicer, ir_slicer)
            if self.spectra_in_ram:
                #indices = self.indices[it_b:it_e+1,ir].tolist()
                indices = self.indices[it_slicer,ir_slicer].tolist()
                print(indices, len(indices))
                indices = h.filter_none_rec(indices)
                print(indices)
                if indices:
                    specZ = h.z2lin(self.spectra[indices,:])
                else:
                    #empty spectrum
                    specZ = np.full((1, 1, self.velocity.shape[0]), h.z2lin(-70))
            else: 
                indices = self.f.variables['spectrum_index'][it_slicer,ir_slicer].tolist()
                print(indices)
                indices = h.filter_none_rec(indices)
                print(indices)
                if indices:
                    specZ = h.z2lin(self.f.variables['radar_power_spectrum_of_copolar_h'][:][indices,:])
                else:
                    #empty spectrum
                    specZ = np.full((1, 1, self.velocity.shape[0]), h.z2lin(-70))

            print('specZ shape', specZ.shape)
            no_averages = np.prod(specZ.shape[:-1])
            specZ = np.average(specZ, axis=(0,1))
            print('specZ shape', specZ.shape)
            assert specZ.shape[0] == self.no_fft, 'no_fft inconsistent'
            specZ = specZ * self.cal_constant_lin * self.range[ir]**2
            specZ_mask = np.logical_or(~np.isfinite(specZ), specZ == 0)
            assert np.all(~specZ_mask), 'mask probably not necessary for kazr spec'
 
            noise = h.estimate_noise(specZ, no_averages) 
            #print("nose_thres {:5.3f} noise_mean {:5.3f} no noise bins {}".format( 
            #    h.lin2z(noise['noise_sep']), h.lin2z(noise['noise_mean']), 
            #    noise['no_noise_bins'])) 
            noise_mean = noise['noise_mean'] 
            #noise_thres = noise['noise_sep'] 
            noise_thres = noise['noise_mean']*peak_finding_params['thres_factor_co']
 
            velocity = self.velocity.copy()
            vel_step = velocity[1] - velocity[0]
            if roll_velocity or ('roll_velocity' in peak_finding_params and peak_finding_params['roll_velocity']):
                if 'roll_velocity' in peak_finding_params and peak_finding_params['roll_velocity']:
                    roll_velocity = peak_finding_params['roll_velocity']
                log.info(f'>> roll velocity active {roll_velocity}')
                upck = _roll_velocity(velocity, vel_step, roll_velocity, 
                                      [specZ, specZ_mask])
                velocity = upck[0]
                specZ, specZ_mask = upck[1]

            window_length = h.round_odd(peak_finding_params['span']/vel_step)
            log.debug(f"window_length {window_length},  polyorder  {peak_finding_params['smooth_polyorder']}")

            specZ_raw = specZ.copy()

            # --------------------------------------------------------------------
            # smoothing and cutting. the order can be defined in instrument_config
            if self.settings['smooth_cut_sequence'] == 'cs':
                specZ[specZ < noise_thres] = np.nan #noise_thres / 6. 
            if peak_finding_params['smooth_polyorder'] != 0:
                specZ = h.lin2z(specZ)
                if peak_finding_params['smooth_polyorder'] < 10:
                    specZ = scipy.signal.savgol_filter(specZ, window_length, 
                                polyorder=peak_finding_params['smooth_polyorder'], 
                                mode='nearest')
                elif 10 < peak_finding_params['smooth_polyorder'] < 20:
                    if indices:
                        _, specZ, _ = loess_1d(velocity, specZ, 
                                    degree=peak_finding_params['smooth_polyorder']-10, 
                                    npoints=window_length)
                elif 20 < peak_finding_params['smooth_polyorder'] < 30:
                    window = h.gauss_func(np.arange(11), 5, window_length)
                    specZ = np.convolve(specZ, window, mode='same')
                else:
                    raise ValueError(f"smooth_polyorder = {peak_finding_params['smooth_polyorder']} not defined")
                specZ = h.z2lin(specZ)
            gaps = (specZ <= 0.) | ~np.isfinite(specZ)
            specZ[gaps] = specZ_raw[gaps]
            if self.settings['smooth_cut_sequence'] == 'sc':
                specZ[specZ < noise_thres] = np.nan #noise_thres / 6. 
            
            specZ_mask = (specZ_mask) | (specZ < noise_thres) | ~np.isfinite(specZ)
 
            specSNRco = specZ/noise_mean 
            specSNRco_mask = specZ.copy() 
 
            peak_finding_params['vel_step'] = vel_step
            #print('Z', h.lin2z(specZ))

            spectrum = { 
                'ts': self.timestamps[it], 'range': self.range[ir],  
                'vel': velocity, 
                'polarimetry': self.settings['polarimetry'],
                'specZ': specZ, 'noise_thres': noise_thres, 
                'specZ_mask': specZ_mask, 
                'specZ_raw': specZ_raw,
                'no_temp_avg': no_averages, 
                'specSNRco': specSNRco, 
                'specSNRco_mask': specSNRco_mask 
            } 

            travtree = generate_tree.tree_from_spectrum_peako(
                #{**spectrum}, peak_finding_params, gaps=gaps) 
                {**spectrum}, peak_finding_params, gaps=None) 
            return travtree, spectrum 

        elif self.type == 'peako':

            log.warning('legacy peako loader, that loades the edges')
            assert temporal_average == False

            specZ = h.z2lin(self.f.variables['spectra'][:,ir,it])
            specZ = specZ*h.z2lin(self.settings['cal_const'])*self.range[ir]**2
            specZ_mask = specZ == 0.
            noise = h.estimate_noise(specZ)
            # some debuggung for the noise estimate

            # for some reason the noise estimate is too high
            #print('raw spec', self.f.variables['spectra'][:20,ir,it])
            #noise_raw_dB = h.estimate_noise(self.f.variables['spectra'][:,ir,it])
            #print("nose_thres {:5.3f} noise_mean {:5.3f}".format(
            #    noise_raw_dB['noise_sep'], noise_raw_dB['noise_mean']))
            #noise_raw_lin = h.estimate_noise(h.z2lin(self.f.variables['spectra'][:,ir,it]))
            #print("nose_thres {:5.3f} noise_mean {:5.3f}".format(
            #    h.lin2z(noise_raw_lin['noise_sep']), h.lin2z(noise_raw_lin['noise_mean'])))

            #print('peako noise level ', self.f.variables['noiselevel'][ir,it])
            #print('calibrated ', h.lin2z(h.z2lin(self.f.variables['noiselevel'][ir,it])*h.z2lin(self.settings['cal_const'])*self.range[ir]**2))
            #noise_thres = noise['noise_sep']
            noise_thres = noise['noise_mean']*peak_finding_params['thres_factor_co']
            noise_mean = noise['noise_mean']

            specSNRco = specZ/noise_mean
            specSNRco_mask = specZ.copy()
            print("nose_thres {:5.3f} noise_mean {:5.3f}".format(
                h.lin2z(noise['noise_sep']), h.lin2z(noise['noise_mean'])))

            spectrum = {
                'ts': self.timestamps[it], 'range': self.range[ir], 
                'vel': self.velocity,
                'polarimetry': self.settings['polarimetry'],
                'specZ': specZ, 'noise_thres': noise_thres,
                'specZ_mask': specZ_mask,
                'no_temp_avg': 0,
                'specSNRco': specSNRco,
                'specSNRco_mask': specSNRco_mask
            }

            left_edges = self.f.variables['left_edge'][:,ir,it].astype(np.int).compressed().tolist()
            right_edges = self.f.variables['right_edge'][:,ir,it].astype(np.int).compressed().tolist()
            bounds = list(zip(left_edges, right_edges))

            if not all([e[0]<e[1] for e in bounds]):
                bounds = []
            d_bounds = h.divide_bounds(bounds)
            print("{} => {} {}".format(bounds, *d_bounds))
            travtree = generate_tree.tree_from_peako({**spectrum}, *d_bounds)
            return travtree, spectrum

        elif self.type == 'rpgpy':
            
            # average_spectra step
            spec_chunk = self.specZ_2d[it_slicer, ir_slicer, :]
            mask_chunk = self.specZ_2d_mask[it_slicer,ir_slicer,:]
            no_averages = np.prod(spec_chunk.shape[:-1])
            specZ = np.average(spec_chunk, axis=(0,1))
            assert not isinstance(specZ, np.ma.core.MaskedArray), "Z not np.ndarray"
            log.debug(f'no_averages {spec_chunk.shape[:-1]} {np.prod(spec_chunk.shape[:-1])}')

            # some parallel processing for debugging
            if self.settings['polarimetry'] == 'STSR':
                spec_cx_chunk = self.specZcx_2d[it_slicer,ir_slicer,:]
                specZcx = np.average(spec_cx_chunk, axis=(0, 1))

                assert not isinstance(specZcx, np.ma.core.MaskedArray), "Zv not np.ndarray"
                #assert not isinstance(specRhv, np.ma.core.MaskedArray), "Rhv not np.ndarray"

            #specLDR = np.average(specLDR_chunk, axis=(0,1))
            mask = np.all(mask_chunk, axis=(0,1))
            log.debug(f'slicer {it_slicer} {ir_slicer} shape {spec_chunk.shape}')
            #log.debug(f'spec shapes {specZ.shape} {specRhv.shape}')
            log.debug(f'spec shapes {specZ.shape}')
            #print('spec_ldr', 10*np.log10(specLDR))

            assert not isinstance(specZ, np.ma.core.MaskedArray), "Z not np.ndarray"
            if np.all(np.isnan(specZ)):
                print('empty spectrum', ir, it)
                return {}, {}

            # TV: noise_v should be more suitable
            # --> update: noise_v actually contains vertical channel noise, better use sum of both channels...?
            #noise_mean = noise_v
            #noise_mean = np.average(self.integ_noise_per_bin[it_slicer, ir_slicer], axis=(0, 1))
            #noise_thres = np.min(specZ[np.isfinite(specZ)])*3
            #noise_thres = noise_h*3
            if ('thres_factor_co' in peak_finding_params
                    and peak_finding_params['thres_factor_co']):
                noise_thres = peak_finding_params['thres_factor_co'] * np.average(self.noise_thres_2d[it_slicer, ir_slicer], axis=(0,1))
            else:
                noise_thres =  np.average(self.noise_thres_2d[it_slicer, ir_slicer], axis=(0,1))

            #ind_chirp = np.where(self.chirp_start_indices >= ir)[0][0] - 1
            #ind_chirp = np.searchsorted(self.chirp_start_indices, ir, side='right')-1
            log.debug(f'current chirp [zero-based index] {ind_chirp}')
            vel_chirp = self.velocity[:, ind_chirp]
            vel_step = vel_chirp[~vel_chirp.mask][1] - vel_chirp[~vel_chirp.mask][0]
            if roll_velocity or ('roll_velocity' in peak_finding_params and peak_finding_params['roll_velocity']):
                if 'roll_velocity' in peak_finding_params and peak_finding_params['roll_velocity']:
                    roll_velocity = peak_finding_params['roll_velocity']
                log.info(f'>> roll velocity active {roll_velocity}')
                if self.settings['polarimetry'] == 'STSR':
                    upck = _roll_velocity(vel_chirp, vel_step, roll_velocity,
                                      #[specZ, specZv, specRhv, mask])
                                      [specZ, specZcx, mask])
                    #specZ, specZv, specRhv, mask = upck[1]
                    specZ, specZcx, mask = upck[1]
                elif self.settings['polarimetry'] == 'false':
                    upck = _roll_velocity(vel_chirp, vel_step, roll_velocity,
                                          [specZ, mask])
                    specZ, mask = upck[1]
                vel_chirp = upck[0]

            # smooth_spectra step
            # TODO: figure out why teresa uses len(velbins) and not /delta_v
            assert 'span' in peak_finding_params, \
                "span and smooth_polyorder have to be defined in config"
            window_length = h.round_odd(peak_finding_params['span']/vel_step)
            log.info(f"span {peak_finding_params['span']} window_length {window_length} polyorder {peak_finding_params['smooth_polyorder']}")

            specZ_raw = specZ.copy()

            # --------------------------------------------------------------------
            # smoothing and cutting. the order can be defined in instrument_config
            if self.settings['smooth_cut_sequence'] == 'cs':
                specZ[specZ < noise_thres] = np.nan #noise_thres / 6. 
            if peak_finding_params['smooth_polyorder'] != 0 and window_length > 1:
                if peak_finding_params['smooth_in_dB']:
                    specZ = h.lin2z(specZ)
                if peak_finding_params['smooth_polyorder'] < 10:
                    specZ = scipy.signal.savgol_filter(specZ, window_length, 
                                polyorder=peak_finding_params['smooth_polyorder'], 
                                mode='nearest')
                elif 10 < peak_finding_params['smooth_polyorder'] < 20:
                    if indices:
                        _, specZ, _ = loess_1d(velocity, specZ, 
                                    degree=peak_finding_params['smooth_polyorder']-10, 
                                    npoints=window_length)
                elif 20 < peak_finding_params['smooth_polyorder'] < 30:
                    window = h.gauss_func(np.arange(11), 5, window_length)
                    specZ = np.convolve(specZ, window, mode='same')
                else:
                    raise ValueError(f"smooth_polyorder = {peak_finding_params['smooth_polyorder']} not defined")
                if peak_finding_params['smooth_in_dB']:
                    specZ = h.z2lin(specZ)
            gaps = (specZ <= 0.) | ~np.isfinite(specZ)
            specZ[gaps] = specZ_raw[gaps]
            if self.settings['smooth_cut_sequence'] == 'sc':
                specZ[specZ < noise_thres] = np.nan #noise_thres / 6. 
            
            # TODO add for other versions
            specZ_mask = (specZ <= 1e-10) | mask | (specZ < noise_thres) | ~np.isfinite(specZ)
            #specZ_mask = (specZ < noise_thres) | ~np.isfinite(specZ)
            # otherwise peak finding identifies fully masked subpeaks
            specZ[specZ_mask] = np.nan  

            peak_finding_params['vel_step'] = vel_step
            #specZ[specZ_mask] = 0

            # also SNR
            if self.settings['polarimetry'] == 'STSR':
                #specZcx_masked = specZcx.copy()
                if ('thres_factor_cx' in peak_finding_params
                        and peak_finding_params['thres_factor_cx']):
                    noise_cx_thres = peak_finding_params['thres_factor_cx'] * np.average(self.noise_thres_2d[it_slicer, ir_slicer], axis=(0,1))
                else:
                    noise_cx_thres =  np.average(self.noise_thres_2d[it_slicer, ir_slicer], axis=(0,1))
                
                specZcx_mask = (specZcx <= 1e-10) | ~np.isfinite(specZcx) | (specZcx < noise_cx_thres)
                log.info(f"noise cx thres {h.lin2z(noise_cx_thres)} {np.all(specZcx_mask)}")
                trust_ldr_mask = specZcx_mask | specZ_mask

                specLDR = (specZcx)/(specZ)
                specLDRmasked = specLDR.copy()
                specLDRmasked[trust_ldr_mask] = np.nan

                #print('specZ', h.lin2z(specLDRmasked))
                #print('specZh', h.lin2z((specZh + specZv)*(1-specRhv)/(specZh + specZv)*(1+specRhv)))

                assert np.isfinite(noise_thres), "noisethreshold is not a finite number"
                spectrum = {
                    'ts': self.timestamps[it], 'range': self.range[ir], 
                    'vel': vel_chirp, 'ind_chirp': ind_chirp,
                    'polarimetry': self.settings['polarimetry'],
                    'specZ': specZ, 'noise_thres': noise_thres,
                    'specZ_mask': specZ_mask,
                    'specZ_raw': specZ_raw,
                    'specZcx': specZcx, 'specZcx_mask': specZcx_mask,
                    #'specZcx_validcx': specZcx_validcx, 'specZ_validcx': specZ_validcx,
                    'no_temp_avg': no_averages,
                    #'specSNRco': specSNRco, 'specSNRco_mask': specSNRco_mask,
                    'noise_cx_thres': noise_cx_thres, 
                    'trust_ldr_mask': trust_ldr_mask,
                    'specLDR': specLDR, 'specLDRmasked': specLDRmasked,
                    #'specZh': specZh, 'cov_re': cov_re, 'cov_im': cov_im, 'rhv': rhv,
                    #'noise_h': noise_h, 'noise_v': noise_v,
                    'decoupling': self.settings['decoupling'],
                }
            elif self.settings['polarimetry'] == 'false':
                specSNRco = specZ/noise_mean
                specSNRco_mask = specZ_mask.copy()
                assert np.isfinite(noise_thres), "noise threshold is not a finite number"
                spectrum = {
                    'ts': self.timestamps[it], 'range': self.range[ir],
                    'vel': vel_chirp,
                    'polarimetry': self.settings['polarimetry'],
                    'specZ': specZ, 'noise_thres': noise_thres,
                    'specZ_mask': specZ_mask,
                    'specZ_raw': specZ_raw,
                    'no_temp_avg': no_averages,
                    'specSNRco': specSNRco, 'specSNRco_mask': specSNRco_mask,
                    #'specZcx_validcx': specZcx_validcx, 'specZ_validcx': specZ_validcx,
                    #'specZh': specZh, 'cov_re': cov_re, 'cov_im': cov_im, 'rhv': rhv,
                    #'noise_h': noise_h, 'noise_v': noise_v,
                    'decoupling': self.settings['decoupling'],
                }


            travtree = {}
            #travtree = generate_tree.tree_from_spectrum({**spectrum}, peak_finding_params)
            travtree = generate_tree.tree_from_spectrum_peako(
                #{**spectrum}, peak_finding_params, gaps=gaps) 
                {**spectrum}, peak_finding_params, gaps=None) 

            return travtree, spectrum


    #@profile
    def assemble_time_height(self, outdir, fname_system=False):
        """convert a whole spectra file to the peakTree node file
        
        Args:
            outdir: directory for the output
        """
        #self.timestamps = self.timestamps[:10]
        with open('output_meta.toml') as output_meta:
            meta_info = toml.loads(output_meta.read())
            if meta_info['contact'] == 'default':
                log.warning('Please specify your contact and institution in output_meta.toml before proceeding!')
                input()

        if self.settings['grid_time']:
            time_grid = get_time_grid(self.timestamps, (self.timestamps[0], self.timestamps[-1]), self.settings['grid_time'])
            timestamps_grid = time_grid[2]
            #print(time_grid)
            #exit()
        else:
            timestamps_grid = self.timestamps
        self.preload_averaging_grid()

        assert self.spectra_in_ram == True
        # self.Z = self.f.variables['Z'][:]
        # self.LDR = self.f.variables['LDR'][:]
        # self.SNRco = self.f.variables['SNRco'][:]

        max_no_nodes=self.settings['max_no_nodes']
        Z = np.zeros((timestamps_grid.shape[0], self.range.shape[0], max_no_nodes))
        Z[:] = -999
        v = Z.copy()
        width = Z.copy()
        skew = Z.copy()
        bound_l = Z.copy()
        bound_r = Z.copy()
        parent = Z.copy()
        thres = Z.copy()
        if self.settings['polarimetry'] in ['STSR', 'LDR']:
            LDR = Z.copy()
            ldrmax = Z.copy()
            ldrmin = Z.copy()
            ldrleft = Z.copy()
            ldrright = Z.copy()
        prominence = Z.copy()
        no_nodes = np.zeros((timestamps_grid.shape[0], self.range.shape[0]))
        part_not_reproduced = np.zeros((timestamps_grid.shape[0], self.range.shape[0]))
        part_not_reproduced[:] = -999

        for it, ts in enumerate(timestamps_grid[:]):
            log.info('it {:5d} ts {}'.format(it, ts))
            it_radar = time_index(self.timestamps, ts)
            log.debug('time {} {} {} radar {} {}'.format(it, h.ts_to_dt(timestamps_grid[it]), timestamps_grid[it], h.ts_to_dt(self.timestamps[it_radar]), self.timestamps[it_radar]))
            #if self.settings['grid_time']:
            #    temp_avg = time_grid[3][it], time_grid[4][it]
            #    log.debug("temp_avg {}".format(temp_avg))

            for ir, rg in enumerate(self.range[:]):
                log.debug("current iteration {} {} {} {}".format(it, h.ts_to_dt(timestamps_grid[it]), ir, rg))
                #log.debug(f"temp average {temp_avg}")
                #travtree, _ = self.get_tree_at(ts, rg, silent=True)
                #if self.settings['grid_time']:
                #    travtree, spectrum = self.get_tree_at((it_radar, self.timestamps[it_radar]), (ir, rg), temporal_average=temp_avg, silent=True)
                #else:
                #    travtree, spectrum = self.get_tree_at((it_radar, self.timestamps[it_radar]), (ir, rg), silent=True)
                # decouple time_grid and averaging
                travtree, spectrum = self.get_tree_at((it_radar, self.timestamps[it_radar]), (ir, rg), silent=True)

                node_ids = list(travtree.keys())
                nodes_to_save = [i for i in node_ids if i < max_no_nodes]
                no_nodes[it,ir] = len(node_ids)
                if spectrum:
                    part_not_reproduced[it,ir] = check_part_not_reproduced(travtree, spectrum)
                else:
                    part_not_reproduced[it,ir] = -999
                #print('max_no_nodes ', max_no_nodes, no_nodes[it,ir], list(travtree.keys()))
                for k in nodes_to_save:
                    if k in travtree.keys() and k < max_no_nodes:
                        val = travtree[k]
                        #print(k,val)
                        Z[it,ir,k] = h.lin2z(val['z'])
                        v[it,ir,k] = val['v']
                        width[it,ir,k] = val['width']
                        skew[it,ir,k] = val['skew']
                        bound_l[it,ir,k] = val['bounds'][0]
                        bound_r[it,ir,k] = val['bounds'][1]
                        parent[it,ir,k] = val['parent_id']
                        thres[it,ir,k] = h.lin2z(val['thres'])
                        prominence[it,ir,k] = h.lin2z(val['prominence'])
                        if self.settings['polarimetry'] in ['STSR', 'LDR']:
                            LDR[it,ir,k] = h.lin2z(val['ldr'])
                            ldrmax[it,ir,k] = h.lin2z(val['ldrmax'])
                            ldrmin[it,ir,k] = h.lin2z(val['ldrmin'])
                            ldrleft[it,ir,k] = h.lin2z(val['ldrleft'])
                            ldrright[it,ir,k] = h.lin2z(val['ldrright'])

        if 'add_to_fname' in self.settings:
            add_to_fname = self.settings['add_to_fname']
            log.info(f'add to fname {add_to_fname}')
        else:
            add_to_fname = ''
        if fname_system:
            sys = f"{self.type}_"
        else:
            sys = ""
        filename = outdir + '{}_{}_{}peakTree{}.nc4'.format(
            self.begin_dt.strftime('%Y%m%d_%H%M'),
            self.shortname, sys, add_to_fname)
        log.info('output filename {}'.format(filename))
        
        with netCDF4.Dataset(filename, 'w', format='NETCDF4') as dataset:
            dim_time = dataset.createDimension('time', Z.shape[0])
            dim_range = dataset.createDimension('range', Z.shape[1])
            dim_nodes = dataset.createDimension('nodes', Z.shape[2])
            dim_vel = dataset.createDimension('vel', self.velocity.shape[0])
            dataset.createDimension('mode', 1)

            print(self.velocity.shape)
            if self.type == 'rpgpy':
                dim_chirp = dataset.createDimension('chirp', self.chirp_start_indices.shape[0])

            times = dataset.createVariable('timestamp', np.int32, ('time',))
            times[:] = timestamps_grid.astype(np.int32)
            times.long_name = 'Unix timestamp [s]'

            dt_list = [h.ts_to_dt(ts) for ts in timestamps_grid]
            hours_cn = np.array([dt.hour + dt.minute / 60. + dt.second / 3600. for dt in dt_list])
            times_cn = dataset.createVariable('time', np.float32, ('time',))
            times_cn[:] = hours_cn.astype(np.float32)
            times_cn.units = "hours since " + self.begin_dt.strftime('%Y-%m-%d') + " 00:00:00 +00:00"
            times_cn.long_name = "Decimal hours from midnight UTC"
            times_cn.axis = "T"

            rg = dataset.createVariable('range', np.float32, ('range',))
            rg[:] = self.range.astype(np.float32)
            rg.long_name = 'range [m]'

            height = self.range + self.settings['station_altitude']
            hg = dataset.createVariable('height', np.float32, ('range',))
            hg[:] = height
            hg.long_name = 'Height above mean sea level [m]'

            if self.type == 'rpgpy':
                vel = dataset.createVariable('velocity', np.float32, ('vel','chirp'))
                vel[:,:] = self.velocity.astype(np.float32)
            else:
                vel = dataset.createVariable('velocity', np.float32, ('vel',))
                vel[:] = self.velocity.astype(np.float32)
            vel.long_name = 'velocity [m/s]'

            saveVar(dataset, {'var_name': 'Z', 'dimension': ('time', 'range', 'nodes'),
                              'arr': Z[:], 'long_name': "Reflectivity factor",
                              #'comment': "",
                              'units': "dBZ", 'missing_value': -999., 'plot_range': (-50., 20.),
                              'plot_scale': "linear"})
            saveVar(dataset, {'var_name': 'v', 'dimension': ('time', 'range', 'nodes'),
                              'arr': v[:], 'long_name': "Velocity",
                              #'comment': "Reflectivity",
                              'units': "m s-1", 'missing_value': -999., 'plot_range': (-8., 8.),
                              'plot_scale': "linear"})
            saveVar(dataset, {'var_name': 'width', 'dimension': ('time', 'range', 'nodes'),
                              'arr': width[:], 'long_name': "Spectral width",
                              #'comment': "Reflectivity",
                              'units': "m s-1", 'missing_value': -999., 'plot_range': (0.01, 3),
                              'plot_scale': "linear"})
            saveVar(dataset, {'var_name': 'skew', 'dimension': ('time', 'range', 'nodes'),
                              'arr': skew[:], 'long_name': "Skewness",
                              #'comment': "",
                              'units': '', 'missing_value': -999., 'plot_range': (-2., 2.),
                              'plot_scale': "linear"})
            saveVar(dataset, {'var_name': 'bound_l', 'dimension': ('time', 'range', 'nodes'),
                              'arr': bound_l[:].astype(np.int32), 'long_name': "Left bound of peak", #'comment': "",
                              'units': "bin", 'missing_value': -999., 'plot_range': (0, self.velocity.shape[0]),
                              'plot_scale': "linear"}, dtype=np.int32)
            saveVar(dataset, {'var_name': 'bound_r', 'dimension': ('time', 'range', 'nodes'),
                              'arr': bound_r[:].astype(np.int32), 'long_name': "Right bound of peak", #'comment': "",
                              'units': "bin", 'missing_value': -999., 'plot_range': (0, self.velocity.shape[0]),
                              'plot_scale': "linear"}, dtype=np.int32)
            saveVar(dataset, {'var_name': 'threshold', 'dimension': ('time', 'range', 'nodes'),
                              'arr': thres[:], 'long_name': "Subpeak Threshold",
                              #'comment': "",
                              'units': "dBZ", 'missing_value': -999., 'plot_range': (-50., 20.),
                              'plot_scale': "linear"})
            saveVar(dataset, {'var_name': 'parent', 'dimension': ('time', 'range', 'nodes'),
                              'arr': parent[:], 'long_name': "Index of the parent node",
                              #'comment': "",
                              'units': "", 'missing_value': -999., 'plot_range': (0, max_no_nodes),
                              'plot_scale': "linear"})
            if self.settings['polarimetry'] in ['STSR', 'LDR']:
                saveVar(dataset, {'var_name': 'decoupling', 'dimension': ('mode'),
                                'arr':  self.settings['decoupling'], 'long_name': "LDR decoupling",
                                'units': "dB", 'missing_value': -999.})
                saveVar(dataset, {'var_name': 'LDR', 'dimension': ('time', 'range', 'nodes'),
                                'arr': LDR[:], 'long_name': "Linear depolarization ratio",
                                #'comment': "",
                                'units': "dB", 'missing_value': -999., 'plot_range': (-25., 0.),
                                'plot_scale': "linear"})
                saveVar(dataset, {'var_name': 'ldrmax', 'dimension': ('time', 'range', 'nodes'),
                                'arr': ldrmax[:], 'long_name': "Maximum LDR from SNR",
                                'units': "", 'missing_value': -999., 'plot_range': (-50., 20.),
                                'plot_scale': "linear"})
                saveVar(dataset, {'var_name': 'ldrmin', 'dimension': ('time', 'range', 'nodes'),
                                'arr': ldrmin[:], 'long_name': "Minimum LDR",
                                'units': "", 'missing_value': -999., 'plot_range': (-50., 20.),
                                'plot_scale': "linear"})
                saveVar(dataset, {'var_name': 'ldrleft', 'dimension': ('time', 'range', 'nodes'),
                                'arr': ldrleft[:], 'long_name': "LDR left of peak center",
                                'units': "", 'missing_value': -999., 'plot_range': (-50., 20.),
                                'plot_scale': "linear"})
                saveVar(dataset, {'var_name': 'ldrright', 'dimension': ('time', 'range', 'nodes'),
                                'arr': ldrright[:], 'long_name': "LDR right of peak center",
                                'units': "", 'missing_value': -999., 'plot_range': (-50., 20.),
                                'plot_scale': "linear"})
            saveVar(dataset, {'var_name': 'prominence', 'dimension': ('time', 'range', 'nodes'),
                              'arr': prominence[:], 'long_name': "Prominence of Peak above threshold",
                              #'comment': "",
                              'units': "", 'missing_value': -999., 'plot_range': (-50., 20.),
                              'plot_scale': "linear"})
            
            saveVar(dataset, {'var_name': 'no_nodes', 'dimension': ('time', 'range'),
                              'arr': no_nodes[:].copy(), 'long_name': "Number of detected nodes",
                              #'comment': "",
                              'units': "", 'missing_value': -999., 'plot_range': (0, max_no_nodes),
                              'plot_scale': "linear"})

            saveVar(dataset, {'var_name': 'part_not_reproduced', 'dimension': ('time', 'range'),
                              'arr': part_not_reproduced[:].copy(), 'long_name': "Part of the spectrum not reproduced by moments",
                              #'comment': "",
                              'units': "m s-1", 'missing_value': -999., 'plot_range': (0, 2),
                              'plot_scale': "linear"})


            dataset.description = 'peakTree processing'
            dataset.location = self.location
            dataset.institution = meta_info["institution"]
            dataset.contact = meta_info["contact"]
            dataset.creation_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
            dataset.settings = str(self.settings)
            dataset.inputinfo = str(self.inputinfo)
            dataset.commit_id = self.git_hash[0]
            dataset.branch = self.git_hash[1]
            dataset.day = str(self.begin_dt.day)
            dataset.month = str(self.begin_dt.month)
            dataset.year = str(self.begin_dt.year)


    def __del__(self):
        if 'f' in list(self.__dict__):
            self.f.close()




if __name__ == "__main__":
    # pTB = peakTreeBuffer()
    # pTB.load_spec_file('../data/D20150617_T0000_0000_Lin_zspc2nc_v1_02_standard.nc4')
    # pTB.get_tree_at(1434574800, 4910)

    pTB = peakTreeBuffer()
    pTB.load_spec_file('../data/D20170311_T2000_2100_Lim_zspc2nc_v1_02_standard.nc4')
    pTB.assemble_time_height('../output/')

    # pTB.load_peakTree_file('../output/20170311_2000_peakTree.nc4')
    # pTB.get_tree_at(1489262634, 3300)

    exit()
    # test the reconstruction:
    ts = 1489262404
    rg = 3300
    #rg = 4000
    #rg = 4100

    pTB = peakTreeBuffer()
    pTB.load_spec_file('../data/D20170311_T2000_2100_Lim_zspc2nc_v1_02_standard.nc4')
    tree_spec, _ = pTB.get_tree_at(ts, rg)
    
    print('del and load new')
    del pTB
    pTB = peakTreeBuffer()
    pTB.load_peakTree_file('../output/20170311_2000_peakTree.nc4')
    tree_file, _ = pTB.get_tree_at(ts, rg)

    print(list(map(lambda elem: (elem[0], elem[1]['coords']), tree_spec.items())))
    print(list(map(lambda elem: (elem[0], elem[1]['coords']), tree_file.items())))
    
    print(list(map(lambda elem: (elem[0], elem[1]['parent_id']), tree_spec.items())))
    print(list(map(lambda elem: (elem[0], elem[1]['parent_id']), tree_file.items())))

    print(list(map(lambda elem: (elem[0], elem[1]['v']), tree_spec.items())))
    print(list(map(lambda elem: (elem[0], elem[1]['v']), tree_file.items())))

