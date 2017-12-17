#! /usr/bin/env python3
# coding=utf-8
"""
Author: radenz@tropos.de


"""

import matplotlib
matplotlib.use('Agg')

import datetime
import netCDF4
import numpy as np
import matplotlib.pyplot as plt

import peakTree.helpers as h

def detect_peak_simple(array, lthres):
    """
    detect noise separated peaks
    """
    ind = np.where(array > lthres)[0].tolist()
    jumps = [ind.index(x) for x, y in zip(ind, ind[1:]) if y - x != 1]
    runs = np.split(ind, [i+1 for i in jumps])
    if runs[0].shape[0] > 0:
        peakindices = [(elem[0], elem[-1]) for elem in runs]
    else:
        peakindices = []
    return peakindices

#@profile
def get_minima(array):
    #sdiff = np.ma.diff(np.sign(np.ma.diff(array)))
    sdiff = np.diff(np.sign(np.diff(array)))
    rising_1 = (sdiff == 2)
    rising_2 = (sdiff[:-1] == 1) & (sdiff[1:] == 1)
    rising_all = rising_1
    rising_all[1:] = rising_all[1:] | rising_2
    min_ind = np.where(rising_all)[0] + 1
    minima = list(zip(min_ind, array[min_ind]))
    return sorted(minima, key=lambda x: x[1])


class Node():
    def __init__(self, bounds, noise_thres, root=False, parent_lvl=0):
        self.bounds = bounds
        self.children = []
        self.level = 0 if root else parent_lvl + 1
        self.root = root
        self.threshold = noise_thres

    def add_noise_sep(self, list_of_bounds, thres):
        for bound in list_of_bounds:
            self.children.append(Node(bound, thres, parent_lvl=self.level))

    def add_min(self, new_index, current_thres):
        if new_index < self.bounds[0] or new_index > self.bounds[1]:
            raise ValueError("child out of parents bounds")
        fitting_child = list(filter(lambda x: x.bounds[0] <= new_index and x.bounds[1] >= new_index, self.children))
        
        if len(fitting_child) == 1:
            fitting_child[0].add_min(new_index, current_thres)
        # or insert here
        else:
            self.children.append(Node((self.bounds[0], new_index), current_thres, parent_lvl=self.level))
            self.children.append(Node((new_index, self.bounds[1]), current_thres, parent_lvl=self.level))
                        
    def __str__(self):
        string = str(self.level) + ' ' + self.level*'  ' + str(self.bounds) + "   [{:4.3e}]".format(self.threshold)
        return "{}\n{}".format(string, ''.join([t.__str__() for t in self.children]))


#@profile
def traverse(Node, coords):
    yield {'coords': coords, 'bounds': Node.bounds, 'thres': Node.threshold}
    for i, n in enumerate(Node.children):
        yield from traverse(n, coords + [i])

#@profile
def coords_to_id(traversed):
    traversed_id = {}   
    level_no = 0
    while True:
        current_level =list(filter(lambda d: len(d['coords']) == level_no+1, traversed))
        if len(current_level) == 0:
            break
        for d in sorted(current_level, key=lambda d: sum(d['coords'])):
            k = len(traversed_id.keys())
            traversed_id[k] = d
            parent = [k for k, val in traversed_id.items() if val['coords'] == d['coords'][:-1]]
            traversed_id[k]['parent_id'] = parent[0] if len(parent) == 1 else -1
        level_no += 1
    return traversed_id

def moment(x, Z):
    mean = np.sum(x*Z)/np.sum(Z)
    rms = np.sqrt(np.sum(((x-mean)**2)*Z)/np.sum(Z))
    skew = np.sum(((x-mean)**3)*Z)/(np.sum(Z)*(rms**3))
    return {'v': mean, 'width': rms, 'skew': skew}

#@profile
def calc_moments(spectrum, bounds, thres):
    """
    calc the moments following the formulas given by Görsdorf2015 and Maahn2017
    """
    Z = np.sum(spectrum['specZ'][bounds[0]:bounds[1]+1])
    # TODO add the masked pocessing for the moments
    #spec_masked = np.ma.masked_less(spectrum['specZ'], thres, copy=True)
    masked_Z = h.fill_with(spectrum['specZ'], spectrum['specZ_mask'], 0.0)
    moments = moment(spectrum['vel'][bounds[0]:bounds[1]+1], masked_Z[bounds[0]:bounds[1]+1])
    
    #spectrum['specZco'] = spectrum['specZ']/(1+spectrum['specLDR'])
    #spectrum['specZcx'] = (spectrum['specLDR']*spectrum['specZ'])/(1+spectrum['specLDR'])
        
    #valid_LDR = h.z2lin(h.lin2z(spectrum['specZcx'].min())+23)
    #masked_specZco = np.ma.masked_less(spectrum['specZco'], valid_LDR, copy=True)
    #masked_specZcx = np.ma.masked_where(masked_specZco.mask, spectrum['specZcx'], copy=True)
    #spectrum['specLDRvalid'] = masked_specZcx/masked_specZco

    ind_max = spectrum['specSNRco'][bounds[0]:bounds[1]+1].argmax()
    ldrmax = spectrum['specSNRcx'][bounds[0]:bounds[1]+1][ind_max]/spectrum['specSNRco'][bounds[0]:bounds[1]+1][ind_max]
    ldrmax_mask = np.logical_or(spectrum['specSNRcx_mask'][bounds[0]:bounds[1]+1][ind_max], 
                                spectrum['specSNRco_mask'][bounds[0]:bounds[1]+1][ind_max])

    #print('ldrmax ', ldrmax, 'ldrmax_mask ', ldrmax_mask)
    moments['ldrmax'] = ldrmax if not isinstance(ldrmax, np.ma.core.MaskedConstant) else spectrum['minSNRcx']/spectrum['specSNRco'][bounds[0]:bounds[1]+1][ind_max]

    #assert np.all(validSNRco.mask == validSNRcx.mask)
    #print('SNRco', h.lin2z(spectrum['validSNRco'][bounds[0]:bounds[1]+1]))
    #print('SNRcx', h.lin2z(spectrum['validSNRcx'][bounds[0]:bounds[1]+1]))
    #print('LDR', h.lin2z(spectrum['validSNRcx'][bounds[0]:bounds[1]+1]/spectrum['validSNRco'][bounds[0]:bounds[1]+1]))
    moments['z'] = Z
    # removed the np.ma. 
    ldr = (np.sum(spectrum['validSNRcx'][bounds[0]:bounds[1]+1])/
        np.sum(spectrum['validSNRco'][bounds[0]:bounds[1]+1]))
    moments['ldr'] = ldr if not isinstance(ldr, np.ma.core.MaskedConstant) else 1e-9

    moments['minv'] = spectrum['vel'][bounds[0]]
    moments['maxv'] = spectrum['vel'][bounds[1]]

    return moments, spectrum

class peakTree():
    """
    -peak detection, tree generation
    -tree compression
    -product generation (payload: Z, v, width, LDR, skew, minv, maxv)
    """

    #@profile
    def get_from_spectrum(self, spectrum):
        """

        spectrum = {'ts': self.timestamps[it], 'range': self.range[ir], 'vel': self.velocity,
            'specZ': specZ, 'noise_thres': specZ.min()}
        """
        smooth = True
        smooth = False
        if smooth:
            print('smoothed spectrum')
            spectrum['specZ'] = np.convolve(spectrum['specZ'], np.array([0.5,1,0.5])/2.0, mode='same')


        masked_Z = h.fill_with(spectrum['specZ'], spectrum['specZ_mask'], 1e-30)
        peak_ind = detect_peak_simple(masked_Z, spectrum['noise_thres'])
        if peak_ind:
            t = Node((0, spectrum['specZ'].shape[0]-1), spectrum['noise_thres'], root=True)
            t.add_noise_sep(peak_ind, spectrum['noise_thres'])
            # minima only inside main peaks
            #minima = get_minima(np.ma.masked_less(spectrum['specZ'], spectrum['noise_thres']*1.1))
            minima = get_minima(np.ma.masked_less(masked_Z, spectrum['noise_thres']*1.1))
            for m in minima:
                #print('minimum ', m)
                t.add_min(m[0], m[1])
            #print(t)
            traversed = coords_to_id(list(traverse(t, [0])))
            for i in traversed.keys():
                if i == 0:
                    moments, spectrum =  calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
                else:
                    moments, _ = calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
                traversed[i].update(moments)
                #print('traversed tree')
                #print(i, traversed[i])

        else:
            traversed = {}
        return traversed



def saveVar(dataset, varData):
    """
    * var_name (Z)
    * dimension ( ('time', 'height') )
    * arr (self.corr_refl_reg[:].filled())
    * long_name ("Reflectivity factor")
    * optional
    * comment ("Wind profiler reflectivity factor corrected by cloud radar (only Bragg contribution)")
    * units ("dBz")
    * missing_value (-200.)
    * plot_range ((-50., 20.))
    * plot_scale ("linear")
    """

    item = dataset.createVariable(varData['var_name'], np.float32, varData['dimension'])
    item[:] = varData['arr']
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


class peakTreeBuffer():
    """
    -read write
    -(time, height, node)

    """
    def load_spec_file(self, filename):
        """load spectra raw file"""
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

        self.begin_dt = h.ts_to_dt(self.timestamps[0])

    def load_peakTree_file(self, filename):
        """load preprocessed peakTree file"""
        self.type = 'peakTree'
        self.f = netCDF4.Dataset(filename, 'r')
        print('loaded file ', filename)
        print('keys ', self.f.variables.keys())
        self.timestamps = self.f.variables['timestamp'][:]
        self.delta_ts = np.mean(np.diff(self.timestamps)) if self.timestamps.shape[0] > 1 else 2.0
        self.range = self.f.variables['range'][:]
        self.no_nodes = self.f.variables['no_nodes'][:]

    #@profile
    def get_tree_at(self, sel_ts, sel_range, silent=False):
        """ """
        it = np.where(self.timestamps == min(self.timestamps, key=lambda t: abs(sel_ts - t)))[0]
        ir = np.where(self.range == min(self.range, key=lambda t: abs(sel_range - t)))[0]
        print('time ', it, h.ts_to_dt(self.timestamps[it]), self.timestamps[it], 'height', ir, self.range[ir]) if not silent else None
        assert np.abs(sel_ts - self.timestamps[it]) < self.delta_ts, 'timestamps more than '+str(self.delta_ts)+'s apart'
        #assert np.abs(sel_range - self.range[ir]) < 10, 'ranges more than 10m apart'

        if self.type == 'spec':
            decoupling = 27

            # why is ravel necessary here?
            # flatten seems to faster
            specZ = self.f.variables['Z'][:,ir,it].ravel()
            specZ_mask = specZ == 0.
            #print('specZ.shape', specZ.shape, specZ)
            specLDR = self.f.variables['LDR'][:,ir,it].ravel()
            specLDR_mask = np.isnan(specLDR)
            specSNRco = self.f.variables['SNRco'][:,ir,it].ravel()
            specSNRco_mask = specSNRco == 0.
            #specSNRco = np.ma.masked_equal(specSNRco, 0)
            noise_thres = 1e-25 if np.all(specZ_mask) else specZ[~specZ_mask].min()
            spectrum = {'ts': self.timestamps[it][0], 'range': self.range[ir][0], 'vel': self.velocity,
                        'specZ': specZ[::-1], 'noise_thres': noise_thres}
            spectrum['specZ_mask'] = specZ_mask[::-1]
            spectrum['specSNRco'] = specSNRco[::-1]
            spectrum['specSNRco_mask'] = specSNRco_mask[::-1]
            spectrum['specLDR'] = specLDR[::-1]
            spectrum['specLDR_mask'] = specLDR_mask[::-1]
            
            spectrum['specZcx'] = spectrum['specZ']*spectrum['specLDR']
            spectrum['specZcx_mask'] = np.logical_or(spectrum['specZ_mask'], spectrum['specLDR_mask'])
            # print('test specZcx calc')
            # print(spectrum['specZcx_mask'])
            # print(spectrum['specZcx'])
            spectrum['specSNRcx'] = spectrum['specSNRco']*spectrum['specLDR']
            spectrum['specSNRcx_mask'] = np.logical_or(spectrum['specSNRco_mask'], spectrum['specLDR_mask'])

            if np.all(spectrum['specSNRcx_mask']):
                spectrum['minSNRcx'] = 1e-99
            else:
                spectrum['minSNRcx'] = spectrum['specSNRcx'][~spectrum['specSNRcx_mask']].min()
            thresSNRcx = spectrum['minSNRcx']*h.z2lin(2.5)
            if np.all(spectrum['specSNRcx_mask']):
                minSNRco = 1e-99
                thresSNRco = 1e-99
                maxSNRco = 1e-99
            else:
                minSNRco = spectrum['specSNRco'][~spectrum['specSNRco_mask']].min()
                thresSNRco = minSNRco*h.z2lin(2.5)
                maxSNRco = minSNRco*h.z2lin(decoupling)

            #validSNRco = np.ma.masked_where((spectrum['specSNRco'] < thresSNRco), spectrum['specSNRco'])
            spectrum['validSNRco_mask'] = np.logical_or(spectrum['specSNRco_mask'], spectrum['specSNRco'] < thresSNRco)

            #validSNRcx = np.ma.masked_where((spectrum['specSNRcx'] < thresSNRcx), spectrum['specSNRcx'])
            spectrum['validSNRcx_mask'] = np.logical_or(spectrum['specSNRcx_mask'], 
                                                        h.fill_with(spectrum['specSNRcx'], spectrum['specSNRcx_mask'], 1e-30) < thresSNRcx)
            #validSNRcx = np.ma.masked_where((spectrum['specSNRco'] > maxSNRco), validSNRcx)
            spectrum['validSNRcx_mask'] = np.logical_or(spectrum['validSNRcx_mask'], 
                                                        h.fill_with(spectrum['specSNRco'], spectrum['specSNRco_mask'], 1e-30) > maxSNRco)
            #validSNRco = np.ma.masked_where(validSNRcx.mask, validSNRco)
            spectrum['validSNRco_mask'] = np.logical_or(spectrum['validSNRcx_mask'], spectrum['validSNRco_mask'])
            
            spectrum['validSNRcx'] = spectrum['specSNRcx'].copy()
            spectrum['validSNRcx'][spectrum['validSNRcx_mask']] = 1e-30
            spectrum['validSNRco'] = spectrum['specSNRco'].copy()
            spectrum['validSNRco'][spectrum['validSNRco_mask']] = 1e-30
            
            spectrum['specLDRmasked'] = spectrum['validSNRcx']/spectrum['validSNRco']
            spectrum['specLDRmasked_mask'] = np.logical_or(spectrum['validSNRcx_mask'], spectrum['validSNRco_mask'])

            trav_tree = peakTree().get_from_spectrum(spectrum)

        elif self.type == 'peakTree':
            print('load tree from peakTree; no_nodes ', self.no_nodes[it,ir])
            trav_tree = {}            
            print('peakTree parent', self.f.variables['parent'][it,ir,:])
            for k in range(int(self.no_nodes[it, ir])):
                print('k', k)
                #(['timestamp', 'range', 'Z', 'v', 'width', 'LDR', 'skew', 'minv', 'maxv', 'threshold', 'parent', 'no_nodes']
                node = {'parent_id': int(np.asscalar(self.f.variables['parent'][it,ir,k])), 
                        'thres': h.z2lin(np.asscalar(self.f.variables['threshold'][it,ir,k])), 
                        'ldr': h.z2lin(np.asscalar(self.f.variables['LDR'][it,ir,k])), 
                        'width': np.asscalar(self.f.variables['width'][it,ir,k]), 
                        #'bounds': self.f.variables[''][it,ir], 
                        'z': h.z2lin(np.asscalar(self.f.variables['Z'][it,ir,k])), 
                        'minv': np.asscalar(self.f.variables['minv'][it,ir,k]), 
                        'maxv': np.asscalar(self.f.variables['maxv'][it,ir,k]), 
                        #'coords': [0], 
                        'skew': np.asscalar(self.f.variables['skew'][it,ir,k]), 
                        'v': np.asscalar(self.f.variables['v'][it,ir,k])}
                if k == 0:
                    node['coords'] = [0]
                else:
                    coords = trav_tree[node['parent_id']]['coords']
                    siblings = list(filter(lambda d: d['coords'][:-1] == coords, trav_tree.values()))
                    print('parent_id', node['parent_id'], 'siblings ', siblings)
                    node['coords'] = coords + [len(siblings)]
                trav_tree[k] = node
      
        return trav_tree

    #@profile
    def assemble_time_height(self, outdir):

        #self.timestamps = self.timestamps[:10]
        max_no_nodes=15
        Z = np.zeros((self.timestamps.shape[0], self.range.shape[0], max_no_nodes))
        Z[:] = -999
        v = Z.copy()
        width = Z.copy()
        LDR = Z.copy()
        skew = Z.copy()
        minv = Z.copy()
        maxv = Z.copy()
        parent = Z.copy()
        thres = Z.copy()
        no_nodes = np.zeros((self.timestamps.shape[0], self.range.shape[0]))

        for it, ts in enumerate(self.timestamps[:]):
            print('time ', it, h.ts_to_dt(self.timestamps[it]), self.timestamps[it])
            for ir, rg in enumerate(self.range[:]):
                trav_tree = self.get_tree_at(ts, rg, silent=True)

                no_nodes[it,ir] = len(list(trav_tree.keys()))

                #print('max_no_nodes ', max_no_nodes, no_nodes[it,ir])
                for k in range(min(max_no_nodes, int(no_nodes[it,ir]))):
                    val = trav_tree[k]
                    #print(k,val)
                    Z[it,ir,k] = h.lin2z(val['z'])
                    v[it,ir,k] = val['v']
                    width[it,ir,k] = val['width']
                    LDR[it,ir,k] = h.lin2z(val['ldr'])
                    skew[it,ir,k] = val['skew']
                    minv[it,ir,k] = val['minv']
                    maxv[it,ir,k] = val['maxv']
                    parent[it,ir,k] = val['parent_id']
                    thres[it,ir,k] = h.lin2z(val['thres'])

        filename = outdir + '{}_peakTree.nc4'.format(self.begin_dt.strftime('%Y%m%d_%H%M'))
        print('output filename ', filename)

        with netCDF4.Dataset(filename, 'w', format='NETCDF4') as dataset:
            dim_time = dataset.createDimension('time', Z.shape[0])
            dim_range = dataset.createDimension('range', Z.shape[1])
            dim_nodes = dataset.createDimension('nodes', Z.shape[2])

            times = dataset.createVariable('timestamp', np.int32, ('time',))
            times[:] = self.timestamps.astype(np.int32)
            times.long_name = 'Unix timestamp [s]'

            rg = dataset.createVariable('range', np.float32, ('range',))
            rg[:] = self.range.astype(np.float32)
            rg.long_name = 'range [m]'

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
            saveVar(dataset, {'var_name': 'LDR', 'dimension': ('time', 'range', 'nodes'),
                              'arr': LDR[:], 'long_name': "Linear depolarization ratio",
                              #'comment': "",
                              'units': "dB", 'missing_value': -999., 'plot_range': (-25., 0.),
                              'plot_scale': "linear"})
            saveVar(dataset, {'var_name': 'skew', 'dimension': ('time', 'range', 'nodes'),
                              'arr': skew[:], 'long_name': "Skewness",
                              #'comment': "",
                              'units': '', 'missing_value': -999., 'plot_range': (-2., 2.),
                              'plot_scale': "linear"})
            saveVar(dataset, {'var_name': 'minv', 'dimension': ('time', 'range', 'nodes'),
                              'arr': minv[:], 'long_name': "Left edge of peak",
                              #'comment': "",
                              'units': "m s-1", 'missing_value': -999., 'plot_range': (-8., 8.),
                              'plot_scale': "linear"})
            saveVar(dataset, {'var_name': 'maxv', 'dimension': ('time', 'range', 'nodes'),
                              'arr': maxv[:], 'long_name': "Right edge of peak",
                              #'comment': "",
                              'units': "m s-1", 'missing_value': -999., 'plot_range': (-8., 8.),
                              'plot_scale': "linear"})
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
            saveVar(dataset, {'var_name': 'no_nodes', 'dimension': ('time', 'range'),
                              'arr': no_nodes[:], 'long_name': "Number of detected nodes",
                              #'comment': "",
                              'units': "", 'missing_value': -999., 'plot_range': (0, max_no_nodes),
                              'plot_scale': "linear"})

            dataset.description = 'peakTree processing'
            dataset.location = 'Limassol'
            dataset.institution = 'TROPOS'
            dataset.contact = 'buehl@tropos.de or radenz@tropos.de'
            dataset.creation_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
            dataset.commit_id = ''
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
    tree_spec = pTB.get_tree_at(ts, rg)
    
    print('del and load new')
    del pTB
    pTB = peakTreeBuffer()
    pTB.load_peakTree_file('../output/20170311_2000_peakTree.nc4')
    tree_file = pTB.get_tree_at(ts, rg)

    print(list(map(lambda elem: (elem[0], elem[1]['coords']), tree_spec.items())))
    print(list(map(lambda elem: (elem[0], elem[1]['coords']), tree_file.items())))
    
    print(list(map(lambda elem: (elem[0], elem[1]['parent_id']), tree_spec.items())))
    print(list(map(lambda elem: (elem[0], elem[1]['parent_id']), tree_file.items())))

    print(list(map(lambda elem: (elem[0], elem[1]['v']), tree_spec.items())))
    print(list(map(lambda elem: (elem[0], elem[1]['v']), tree_file.items())))