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
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from . import helpers as h
from . import print_tree
from . import generate_tree
import toml

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
    delta_v = spectrum['vel'][2] - spectrum['vel'][1]
    
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
    label = subprocess.check_output(['git', 'describe', '--always'])
    return label.rstrip()


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

        self.peak_finding_params = config[system]['settings']['peak_finding_params']
        
        self.git_hash = get_git_hash()


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
        self.velocity = self.f.variables['velocity_bins'][:] 

        self.cal_constant = float(self.f.cal_constant[:-3])
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

    def load_peako_file(self, filename):
        """

        """
        self.type = 'peako'
        self.f = netCDF4.Dataset(filename, 'r')
        print('loaded file ', filename)
        print('keys ', self.f.variables.keys())
        #times = self.f.variables['decimal_time'][:]
        #seconds = times.astype(np.float)*3600.
        #self.timestamps = h.dt_to_ts(datetime.datetime(2014, 2, 21)) + seconds
        self.timestamps = self.f.variables['unix_time'][:]
        self.delta_ts = np.mean(np.diff(self.timestamps)) if self.timestamps.shape[0] > 1 else 2.0
        self.range = self.f.variables['range'][:]*1000
        self.velocity = self.f.variables['Dopplerbins'][:]

        self.begin_dt = h.ts_to_dt(self.timestamps[0])


    def load_joyrad_file(self, filename, load_to_ram=False):
        """load spectra from raw file
        
        Args:
            filename: specify file
        """
        self.type = 'joyrad'

        self.f = netCDF4.Dataset(filename, 'r')
        print('keys ', self.f.variables.keys())

        #print(self.f)
        self.timestamps = self.f.variables['time'][:].astype(np.float64) + self.f.variables['sampleTms'][:]/1000.
        self.timestamps += h.dt_to_ts(datetime.datetime(2001, 1, 1, 0, 0, 0))
        print('time ', self.timestamps[:10])
        self.delta_ts = np.mean(np.diff(self.timestamps)) if self.timestamps.shape[0] > 1 else 2.0
        self.range = self.f.variables['range'][:]
        print('range ', self.range[:10])
        self.minvel = self.f.variables['MinVel'][:]
        #self.shiftvel = self.f.variables['MinVel_Correction'][:]

        self.ir_chirps = self.f.variables['range_offsets'][:] - 1
        self.chirp_vel = self.f.variables['velocity'][:]
        self.velocity = np.zeros(512)
#        print('velocity ', self.velocity[:10])
        print('Z chunking ', self.f.variables['spec'].chunking())

        self.begin_dt = h.ts_to_dt(self.timestamps[0])

        if load_to_ram == True:
            self.spectra_in_ram = True
            self.spec = self.f.variables['spec'][:]
        #     self.LDR = self.f.variables['LDR'][:]
        #     self.SNRco = self.f.variables['SNRco'][:]



    #@profile
    def get_tree_at(self, sel_ts, sel_range, temporal_average=False, 
                    roll_velocity=False, peak_finding_params={}, silent=False):
        """get a single tree
        either from the spectrum directly (prior call of load_spec_file())
        or from the pre-converted file (prior call of load_peakTree_file())

        Args:
            sel_ts (tuple or float): either value or (index, value)
            sel_range (tuple or float): either value or (index, value)
            temporal_average (optional): aberage over the interval 
                (tuple of bin in time dimension or number of seconds)
            roll_velocity (optional): shift the x rightmost bins to the
                left 
            peak_finding_params (optional):
            silent: verbose output

        'vel_smooth': convolution with given array 
        'prom_thres': prominence threshold in dB
        
        Returns:
            dictionary with all nodes and the parameters of each node
        """

        peak_finding_params = (lambda d: d.update(peak_finding_params) or d)(self.peak_finding_params)
        # print('using peak_finding_params', peak_finding_params)

        if type(sel_ts) is tuple and type(sel_range) is tuple:
            it, sel_ts = sel_ts
            ir, sel_range = sel_range
        else:
            it = time_index(self.timestamps, sel_ts)
            ir = np.where(self.range == min(self.range, key=lambda t: abs(sel_range - t)))[0][0]
        log.info('time {} {} {} height {} {}'.format(it, h.ts_to_dt(self.timestamps[it]), self.timestamps[it], ir, self.range[ir])) if not silent else None
        assert np.abs(sel_ts - self.timestamps[it]) < self.delta_ts, 'timestamps more than '+str(self.delta_ts)+'s apart'
        #assert np.abs(sel_range - self.range[ir]) < 10, 'ranges more than 10m apart'

        if type(temporal_average) is tuple:
            it_b, it_e = temporal_average
        elif temporal_average:
            it_b = time_index(self.timestamps, sel_ts-temporal_average/2.)
            it_e = time_index(self.timestamps, sel_ts+temporal_average/2.)
            assert self.timestamps[it_e] - self.timestamps[it_b] < 15, 'found averaging range too large'
            log.debug('timerange {} {} {} {}'.format(it_b, h.ts_to_dt(self.timestamps[it_b]), it_e, h.ts_to_dt(self.timestamps[it_e]))) if not silent else None

        if self.type == 'spec':
            decoupling = self.settings['decoupling']

            # why is ravel necessary here?
            # flatten seems to faster
            if not temporal_average:
                if self.spectra_in_ram:
                    specZ = self.Z[:,ir,it].ravel()
                    specLDR = self.LDR[:,ir,it].ravel()
                    specSNRco = self.SNRco[:,ir,it].ravel()
                else:
                    specZ = self.f.variables['Z'][:,ir,it].ravel()
                    specLDR = self.f.variables['LDR'][:,ir,it].ravel()
                    specSNRco = self.f.variables['SNRco'][:,ir,it].ravel()
                specZ_mask = specZ == 0.
                #print('specZ.shape', specZ.shape, specZ)
                
                specLDR_mask = np.isnan(specLDR)
                specZcx = specZ*specLDR
                specZcx_mask = np.logical_or(specZ_mask, specLDR_mask)
                
                specSNRco_mask = specSNRco == 0.
                no_averages = 1
            else:
                if self.spectra_in_ram:
                    #specZ = self.f.variables['Z'][:,ir,it_b:it_e+1]
                    specZ_2d = self.Z[:,ir,it_b:it_e+1][:]
                    no_averages = specZ_2d.shape[1]
                    specLDR_2d = self.LDR[:,ir,it_b:it_e+1][:]
                    specZcx_2d = specZ_2d*specLDR_2d
                    # np.average is slightly faster than .mean
                    #specZcx = specZcx_2d.mean(axis=1)
                    specZcx = np.average(specZcx_2d, axis=1)
                    #specZ = specZ_2d.mean(axis=1)
                    specZ = np.average(specZ_2d, axis=1)
                    specLDR = specZcx/specZ
                    #specSNRco_2d = self.SNRco[:,ir,it_b:it_e+1][:]
                    #specSNRco = specSNRco_2d.mean(axis=1)
                    #specSNRco = np.average(specSNRco_2d, axis=1)
                    # specZ, specZcx, specLDR, specSNRco, no_averages = fast_funcs.load_spec_mira(
                    #     self.Z, self.LDR, self.SNRco, ir, it_b, it_e)
                    assert not isinstance(specZ, np.ma.core.MaskedArray), "Z not np.ndarray"
                    assert not isinstance(specZcx, np.ma.core.MaskedArray), "Z not np.ndarray"
                    assert not isinstance(specLDR, np.ma.core.MaskedArray), "LDR not np.ndarray"
                else:
                    specZ = self.f.variables['Z'][:,ir,it_b:it_e+1][:]
                    no_averages = specZ.shape[1]
                    specLDR = self.f.variables['LDR'][:,ir,it_b:it_e+1][:]
                    specZcx = specZ*specLDR
                    specZcx = specZcx.mean(axis=1)
                    specZ = specZ.mean(axis=1)

                    specLDR = specZcx/specZ
                    #specSNRco = self.f.variables['SNRco'][:,ir,it_b:it_e+1][:]
                    #specSNRco = specSNRco.mean(axis=1)

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
            noise_thres = 1e-25 if empty_spec else np.min(specZ[~specZ_mask])*h.z2lin(peak_finding_params['thres_factor_co'])
            
            if roll_velocity or ('roll_velocity' in self.settings and self.settings['roll_velocity']):
                if 'roll_velocity' in self.settings and self.settings['roll_velocity']:
                    roll_velocity = self.settings['roll_velocity']
                vel_step = self.velocity[1] - self.velocity[0]
                self.velocity = np.concatenate((
                    np.linspace(self.velocity[0] - roll_velocity * vel_step, 
                                self.velocity[0] - vel_step, 
                                num=roll_velocity), 
                    self.velocity[:-roll_velocity]))

                specZ = np.concatenate((specZ[-roll_velocity:], 
                                        specZ[:-roll_velocity]))
                specLDR = np.concatenate((specLDR[-roll_velocity:], 
                                          specLDR[:-roll_velocity]))
                specZcx = np.concatenate((specZcx[-roll_velocity:], 
                                          specZcx[:-roll_velocity]))
                specZ_mask = np.concatenate((specZ_mask[-roll_velocity:], 
                                             specZ_mask[:-roll_velocity]))
                specZcx_mask = np.concatenate((specZcx_mask[-roll_velocity:], 
                                               specZcx_mask[:-roll_velocity]))
                specLDR_mask = np.concatenate((specLDR_mask[-roll_velocity:], 
                                               specLDR_mask[:-roll_velocity]))

            if 'span' in peak_finding_params:
                window_length = h.round_odd(peak_finding_params['span']/(self.velocity[1]-self.velocity[0]))
                log.debug(f"window_length {window_length},  polyorder  {peak_finding_params['smooth_polyorder']}")
                specZ = scipy.signal.savgol_filter(specZ, window_length, polyorder=1, mode='nearest')
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

            
            spectrum = {'ts': self.timestamps[it], 'range': self.range[ir],
                        'noise_thres': noise_thres, 'no_temp_avg': no_averages}

            spectrum['specZ'] = specZ[:]
            spectrum['vel'] = self.velocity
            
            spectrum['specZ_mask'] = specZ_mask[:]
            #spectrum['specSNRco'] = specSNRco[:]
            #spectrum['specSNRco_mask'] = specSNRco_mask[:]
            spectrum['specLDR'] = specLDR[:]
            spectrum['specLDR_mask'] = specLDR_mask[:]
            
            spectrum['specZcx'] = specZcx[:]
            # spectrum['specZcx_mask'] = np.logical_or(spectrum['specZ_mask'], spectrum['specLDR_mask'])
            spectrum['specZcx_mask'] =  specZcx_mask[:]

            # print('test specZcx calc')
            # print(spectrum['specZcx_mask'])
            # print(spectrum['specZcx'])
            #spectrum['specSNRcx'] = spectrum['specSNRco']*spectrum['specLDR']
            #spectrum['specSNRcx_mask'] = np.logical_or(spectrum['specSNRco_mask'], spectrum['specLDR_mask'])

            thres_Zcx = np.nanmin(spectrum['specZcx'])*h.z2lin(peak_finding_params['thres_factor_cx'])
            Zcx_mask = np.logical_or(spectrum['specZcx'] < thres_Zcx, ~np.isfinite(spectrum['specZcx']))
            spectrum['specLDRmasked'] = spectrum['specLDR'].copy()
            spectrum['specLDRmasked'][Zcx_mask] = np.nan

            spectrum['specZcx_mask'] = Zcx_mask
            spectrum['specZcx_validcx'] = spectrum['specZcx'].copy()
            spectrum['specZcx_validcx'][Zcx_mask] = 0.0
            spectrum['specZ_validcx'] = spectrum['specZ'].copy()
            spectrum['specZ_validcx'][Zcx_mask] = 0.0

            spectrum['decoupling'] = decoupling

            #                                     deep copy of dict 
            travtree = generate_tree.tree_from_spectrum({**spectrum}, peak_finding_params)
            
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
 
            if roll_velocity or ('roll_velocity' in self.settings and self.settings['roll_velocity']):
                if 'roll_velocity' in self.settings and self.settings['roll_velocity']:
                    roll_velocity = self.settings['roll_velocity']
                vel_step = self.velocity[1] - self.velocity[0]
                self.velocity = np.concatenate((
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
                specZ = scipy.signal.savgol_filter(specZ, window_length, polyorder=1, mode='nearest')
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
                'vel': self.velocity, 
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
             
            if not temporal_average:
                if self.spectra_in_ram:
                    index = self.indices[it,ir]
                    specZ = h.z2lin(self.spectra[index,:])
                else:
                    index = self.f.variables['spectrum_index'][it,ir]
                    specZ = h.z2lin(self.f.variables['radar_power_spectrum_of_copolar_h'][index,:])
                    # specZ = self.f.variables['spectra'][it,ir,:] 
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
                    indices = self.f.variables['spectrum_index'][it_b:it_e+1,ir].tolist()
                    indices = [i for i in indices if i is not None]
                    if indices:
                        specZ = h.z2lin(self.f.variables['radar_power_spectrum_of_copolar_h'][indices,:])
                    else:
                        #empty spectrum
                        specZ = np.full((2, self.velocity.shape[0]), h.z2lin(-70))

                no_averages = specZ.shape[0] 
                specZ = np.average(specZ, axis=0)
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
 
            if roll_velocity or ('roll_velocity' in self.settings and self.settings['roll_velocity']):
                if 'roll_velocity' in self.settings and self.settings['roll_velocity']:
                    roll_velocity = self.settings['roll_velocity']
                vel_step = self.velocity[1] - self.velocity[0]
                self.velocity = np.concatenate((
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
                specZ = scipy.signal.savgol_filter(specZ, window_length, polyorder=1, mode='nearest')
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
 
            #print('Z', h.lin2z(specZ))

            spectrum = { 
                'ts': self.timestamps[it], 'range': self.range[ir],  
                'vel': self.velocity, 
                'specZ': specZ, 'noise_thres': noise_thres, 
                'specZ_mask': specZ_mask, 
                'no_temp_avg': no_averages, 
                'specSNRco': specSNRco, 
                'specSNRco_mask': specSNRco_mask 
            } 

            travtree = generate_tree.tree_from_spectrum({**spectrum}, peak_finding_params) 
            return travtree, spectrum 


        elif self.type == 'peako':
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
                'specZ': specZ, 'noise_thres': noise_thres,
                'specZ_mask': specZ_mask,
                'no_temp_avg': 0,
                'specSNRco': specSNRco,
                'specSNRco_mask': specSNRco_mask
            }

            def divide_bounds(bounds):
                """
                divide_bounds([[10,20],[20,25],[25,30],[40,50]]) 
                => noise_sep [[10, 30], [40, 50]], internal [20, 25]
                """
                bounds = list(sorted(h.flatten(bounds)))
                occ = dict((k, (bounds).count(k)) for k in set(bounds))
                internal = [k for k in occ if occ[k] == 2]
                noise_sep = [b for b in bounds if b not in internal]
                noise_sep = [[noise_sep[i], noise_sep[i+1]] for i in \
                             range(0, len(noise_sep)-1,2)]
                return noise_sep, internal

            left_edges = self.f.variables['left_edge'][:,ir,it].astype(np.int).compressed().tolist()
            right_edges = self.f.variables['right_edge'][:,ir,it].astype(np.int).compressed().tolist()
            bounds = list(zip(left_edges, right_edges))

            if not all([e[0]<e[1] for e in bounds]):
                bounds = []
            d_bounds = divide_bounds(bounds)
            print("{} => {} {}".format(bounds, *d_bounds))
            travtree = generate_tree.tree_from_peako({**spectrum}, *d_bounds)
            return travtree, spectrum

        elif self.type == 'joyrad':

            #import scipy.stats
            #print('minvel ', self.minvel[it, ir])
            #self.velocity = np.linspace(self.minvel[it, ir], -self.minvel[it, ir], num=)
            if np.isnan(self.minvel[it, ir]):
                return {}, {}

            # TODO remove the hardcoded number of bins

            def get_chirp(ir_chirps, ir):
                if ir >= ir_chirps[-1]:
                    return ir_chirps.shape[0] - 1
                else:
                    return max(0, np.argwhere(self.ir_chirps > ir)[0] - 1)

            if not temporal_average:
                if self.spectra_in_ram:
                    specZ = self.spec[it,ir]
                else:
                    specZ = self.f.variables['spec'][it,ir]
                no_averages = 1 
                specZ_mask = specZ == 0. 

                #print(ir, self.ir_chirps, get_chirp(self.ir_chirps, ir))
                velocity = self.chirp_vel[get_chirp(self.ir_chirps, ir)].ravel()
                #velocity = np.linspace(self.minvel[it, ir], -self.minvel[it, ir], num=512) + self.shiftvel[it,ir]
                if not np.isnan(self.minvel[it,ir]):
                    velocity = velocity + self.minvel[it,ir] - velocity[0]

            else:
                
                
                if self.spectra_in_ram:
                    print('it_b it_e', it_b, it_e, ir)

                    # all_vels = np.array([])
                    # all_spec = np.array([])
                    # max_shape = self.minvel.shape

                    # for j in range(it_b, it_e+1):
                    #     velocity = self.chirp_vel[get_chirp(self.ir_chirps, ir)].ravel()
                    #     #print(velocity, np.mean(velocity))
                    #     if not np.isnan(self.minvel[min(j,max_shape[1]-1),ir]):
                    #         velocity = velocity + self.minvel[min(j,max_shape[1]-1),ir] - velocity[0]

                    #     all_vels = np.append(
                    #         all_vels, velocity)
                    #     all_spec = np.append(
                    #         all_spec, self.spec[min(j,max_shape[1]-1),ir])
                    #print(all_vels.shape, all_vels)
                    #print(all_spec.shape, all_spec)
                    # 
                    # average on non-similar velocity grid.
                    #
                    # delta_vel = all_vels[~np.isnan(all_vels)][1] - all_vels[~np.isnan(all_vels)][0]
                    # #print(delta_vel, np.all(np.isnan(all_vels)))
                    # specZ, edges, binno = scipy.stats.binned_statistic(all_vels, all_spec, 'mean', bins=np.arange(-15,10,delta_vel*2))
                    # velocity = edges[:-1] + delta_vel*2
                    #print(velocity.shape, specZ.shape, edges.shape)

                    specZ_2d = self.spec[it_b:it_e+1,ir,:][:]
                    no_averages = specZ_2d.shape[1]
                    specZ = specZ_2d.mean(axis=0)
                    #print('specZ_2d.shape', specZ_2d.shape, '=>', specZ.shape)
                    velocity = self.chirp_vel[get_chirp(self.ir_chirps, ir)].ravel()
                    #specSNRco = self.f.variables['SNRco'][:,ir,it_b:it_e+1]

                else: 

                    # raise ValueError('implementation changed')
                    
                    print('it_b it_e', it_b, it_e)
                    specZ_2d = self.f.variables['spec'][it_b:it_e+1,ir,:][:]
                    no_averages = specZ_2d.shape[0]
                    specZ = specZ_2d.mean(axis=0)
                    #print('specZ_2d.shape', specZ_2d.shape, '=>', specZ.shape)
                    velocity = self.chirp_vel[get_chirp(self.ir_chirps, ir)].ravel()

                    # all_vels = np.array([])
                    # all_spec = np.array([])

                    # for j in range(it_b, it_e+1):
                    #     velocity = self.chirp_vel[get_chirp(self.ir_chirps, ir)].ravel()
                    #     velocity = velocity + self.minvel[j,ir] - velocity[0]

                    #     all_vels = np.append(
                    #         all_vels, velocity)
                    #     all_spec = np.append(
                    #         all_spec, self.f.variables['spec'][j,ir])
                    # #print(all_vels.shape, all_vels)
                    # #print(all_spec.shape, all_spec)
                    # # 
                    # # average on non-similar velocity grid.
                    # #
                    # delta_vel = all_vels[1]-all_vels[0]
                    # specZ, edges, binno = scipy.stats.binned_statistic(all_vels, all_spec, 'mean', bins=np.arange(-15,10,delta_vel*2))
                    # velocity = edges[:-1] + delta_vel*2
                    # print(velocity.shape, specZ.shape, edges.shape)

                    # else:
                    #     #empty spectrum
                    #     specZ = np.full((2, self.velocity.shape[0]), h.z2lin(-70))

                no_averages = it_e - it_b + 1
                specZ_mask = specZ == 0. 
 
            noise = h.estimate_noise(specZ, no_averages) 
            log.debug("nose_thres {:5.3f} noise_mean {:5.3f} no noise bins {}".format( 
               h.lin2z(noise['noise_sep']), h.lin2z(noise['noise_mean']), 
               noise['no_noise_bins'])) 
            noise_mean = noise['noise_mean'] 
            #noise_thres = noise['noise_sep'] 
            noise_thres = noise['noise_mean']*peak_finding_params['thres_factor_co']
 
            # if self.settings['smooth'] and self.settings['smooth'] != 'broad': 
            #     #print('smoothed spectrum small') 
            #     specZ = np.convolve(specZ,  
            #                         np.array([0.5,0.7,1,0.7,0.5])/3.4,  
            #                         mode='same')
            # elif self.settings['smooth'] and self.settings['smooth'] == 'broad': 
            #     #print('smoothed spectrum broad') 
            #     specZ = np.convolve(specZ,  
            #                         np.array([0.5,0.7,0.8,0.9,1,1,1,0.9,0.8,0.7,0.5])/8.8,  
            #                         mode='same')
            if np.any(peak_finding_params['vel_smooth']):
                #print('smoothed spectrum')
                if isinstance(peak_finding_params['vel_smooth'],(list,np.ndarray)):
                    convol_window = peak_finding_params['vel_smooth']
                else:
                    if peak_finding_params['vel_smooth'] != 'broad':
                        convol_window = np.array([0.5,0.7,1,0.7,0.5])/3.4
                    elif peak_finding_params['vel_smooth'] == 'broad':
                        convol_window = np.array([0.5,0.7,0.8,0.9,1,1,1,0.9,0.8,0.7,0.5])/8.8
                specZ = np.convolve(specZ,  
                                    convol_window,
                                    mode='same') 
 
            specSNRco = specZ/noise_mean 
            specSNRco_mask = specZ.copy() 
 
 
            spectrum = { 
                'ts': self.timestamps[it], 'range': self.range[ir],  
                'vel': velocity, 
                'specZ': specZ, 'noise_thres': noise_thres, 
                'specZ_mask': specZ_mask, 
                'no_temp_avg': no_averages, 
                'specSNRco': specSNRco, 
                'specSNRco_mask': specSNRco_mask 
            } 
 
            if roll_velocity or ('roll_velocity' in self.settings and self.settings['roll_velocity']):
                if 'roll_velocity' in self.settings and self.settings['roll_velocity']:
                    roll_velocity = self.settings['roll_velocity']
                vel_step = self.velocity[1] - self.velocity[0]
                spectrum['vel'] = np.concatenate((
                    np.linspace(self.velocity[0] - roll_velocity * vel_step, 
                                self.velocity[0] - vel_step, 
                                num=roll_velocity), 
                    self.velocity[:-roll_velocity]))
                keys_to_roll = ['specZ', 'specZ_mask', 'specSNRco', 'specSNRco_mask']
                for k in keys_to_roll:
                    spectrum[k] = np.concatenate((
                        spectrum[k][-roll_velocity:], 
                        spectrum[k][:-roll_velocity]))

            travtree = generate_tree.tree_from_spectrum({**spectrum}, peak_finding_params) 
            return travtree, spectrum 

    #@profile
    def assemble_time_height(self, outdir):
        """convert a whole spectra file to the peakTree node file
        
        Args:
            outdir: directory for the output
        """
        #self.timestamps = self.timestamps[:10]

        if self.settings['grid_time']:
            time_grid = get_time_grid(self.timestamps, (self.timestamps[0], self.timestamps[-1]), self.settings['grid_time'])
            timestamps_grid = time_grid[2]
        else:
            timestamps_grid = self.timestamps

        assert self.spectra_in_ram == True
        # self.Z = self.f.variables['Z'][:]
        # self.LDR = self.f.variables['LDR'][:]
        # self.SNRco = self.f.variables['SNRco'][:]

        max_no_nodes=self.settings['max_no_nodes']
        Z = np.zeros((timestamps_grid.shape[0], self.range.shape[0], max_no_nodes))
        Z[:] = -999
        v = Z.copy()
        width = Z.copy()
        LDR = Z.copy()
        skew = Z.copy()
        bound_l = Z.copy()
        bound_r = Z.copy()
        parent = Z.copy()
        thres = Z.copy()
        ldrmax = Z.copy()
        prominence = Z.copy()
        no_nodes = np.zeros((timestamps_grid.shape[0], self.range.shape[0]))
        part_not_reproduced = np.zeros((timestamps_grid.shape[0], self.range.shape[0]))
        part_not_reproduced[:] = -999

        for it, ts in enumerate(timestamps_grid[:]):
            log.info('it {:5d} ts {}'.format(it, ts))
            it_radar = time_index(self.timestamps, ts)
            log.debug('time {} {} {} radar {} {}'.format(it, h.ts_to_dt(timestamps_grid[it]), timestamps_grid[it], h.ts_to_dt(self.timestamps[it_radar]), self.timestamps[it_radar]))
            if self.settings['grid_time']:
                temp_avg = time_grid[3][it], time_grid[4][it]
                log.debug("temp_avg {}".format(temp_avg))

            for ir, rg in enumerate(self.range[:]):
                log.debug("current iteration {} {} {}".format(h.ts_to_dt(timestamps_grid[it]), ir, rg))
                #travtree, _ = self.get_tree_at(ts, rg, silent=True)
                if self.settings['grid_time']:
                    travtree, spectrum = self.get_tree_at((it_radar, self.timestamps[it_radar]), (ir, rg), temporal_average=temp_avg, silent=True)
                else:
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
                        LDR[it,ir,k] = h.lin2z(val['ldr'])
                        skew[it,ir,k] = val['skew']
                        bound_l[it,ir,k] = val['bounds'][0]
                        bound_r[it,ir,k] = val['bounds'][1]
                        parent[it,ir,k] = val['parent_id']
                        thres[it,ir,k] = h.lin2z(val['thres'])
                        ldrmax[it,ir,k] = h.lin2z(val['ldrmax'])
                        prominence[it,ir,k] = h.lin2z(val['prominence'])

        if 'add_to_fname' in self.settings:
            add_to_fname = self.settings['add_to_fname']
            log.info(f'add to fname {add_to_fname}')
        else:
            add_to_fname = ''
        filename = outdir + '{}_{}_peakTree{}.nc4'.format(self.begin_dt.strftime('%Y%m%d_%H%M'),
                                                          self.shortname, add_to_fname)
        log.info('output filename {}'.format(filename))
        
        with netCDF4.Dataset(filename, 'w', format='NETCDF4') as dataset:
            dim_time = dataset.createDimension('time', Z.shape[0])
            dim_range = dataset.createDimension('range', Z.shape[1])
            dim_nodes = dataset.createDimension('nodes', Z.shape[2])
            dim_nodes = dataset.createDimension('vel', self.velocity.shape[0])
            dataset.createDimension('mode', 1)

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
            if self.settings['LDR']:
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
                                #'comment': "",
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

            with open('output_meta.toml') as output_meta:
                meta_info = toml.loads(output_meta.read())

            dataset.description = 'peakTree processing'
            dataset.location = self.location
            dataset.institution = meta_info["institution"]
            dataset.contact = meta_info["contact"]
            dataset.creation_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
            dataset.settings = str(self.settings)
            dataset.commit_id = self.git_hash
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
