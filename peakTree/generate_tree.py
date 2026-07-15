""""""
"""
Author: radenz@tropos.de
"""

import logging
import numpy as np
from . import helpers as h
from . import print_tree
from numba import jit
import copy
import xarray as xr

import scipy.signal

log = logging.getLogger('peakTree')

#@profile
@jit(fastmath=True)
def detect_peak_simple(array, lthres):
    """detect noise separated peaks

    Args:
        array: with Doppler spectrum
        lthres: threshold
    Returns:
        list of indices (as tuple)
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
@jit(nopython=True, fastmath=True)
def get_minima(array):
    """get the minima of an array by calculating the derivative

    tested against scipy.signal.argrelmin without difference
    in result or speed

    Returns:
        list of ``(index, value at index)``
    """
    #sdiff = np.ma.diff(np.sign(np.ma.diff(array)))
    sdiff = np.diff(np.sign(np.diff(array)))
    rising_1 = (sdiff == 2)
    rising_2 = (sdiff[:-1] == 1) & (sdiff[1:] == 1)
    rising_all = rising_1
    rising_all[1:] = rising_all[1:] | rising_2
    min_ind = np.where(rising_all)[0] + 1
    minima = list(zip(min_ind, array[min_ind]))
    # numba jit and itemgetter are not compatible
    return sorted(minima, key=lambda x: x[1])
    #return sorted(minima, key=itemgetter(1))


# @jit(fastmath=True)
# def split_peak_ind_by_space_legacy(peak_ind):
#     """split a list of peak indices by their maximum space
#     use for noise floor separated peaks
#     
#     Args:
#         peak_ind: list of peak indices ``[(163, 165), (191, 210), (222, 229), (248, 256)]``
#     Returns:
#         left sublist, right sublist
#     """
#     if len(peak_ind) == 1:
#         return [peak_ind, peak_ind]
#     left_i = np.array([elem[0] for elem in peak_ind])
#     right_i = np.array([elem[1] for elem in peak_ind])
#     spacing = left_i[1:]-right_i[:-1]
#     split_i = np.argmax(spacing)
#     return peak_ind[:split_i+1], peak_ind[split_i+1:]


def split_peak_ind_by_space(peak_ind):
    """split a list of peak indices by their maximum space
    use for noise floor separated peaks
    
    Args:
        peak_ind: list of peak indices ``[(163, 165), (191, 210), (222, 229), (248, 256)]``
    Returns:
        left sublist, right sublist
    """
    if len(peak_ind) == 1:
        return peak_ind, peak_ind
    p_ind = np.array(peak_ind)
    spacing = p_ind[:,0][1:]-p_ind[:,1][:-1]
    split_i = np.argmax(spacing)
    return p_ind[:split_i+1].tolist(), p_ind[split_i+1:].tolist()


def peak_pairs_to_call(peak_ind):
    """generator that yields the tree structure of a peak list based on spacing
    
    Args:
        peak_ind: list of peak indices
    Yields:
        tree structure for noise separated peaks (includes recursive ``yield from`` for children)
    """
    left, right = split_peak_ind_by_space(peak_ind)
    if left != right:
        yield (left[0][0], left[-1][-1]), (right[0][0], right[-1][-1])
        yield from peak_pairs_to_call(left)
        yield from peak_pairs_to_call(right)


class Node():
    """class to generate the tree
    
    Args:
        bounds: boundaries in bin coordinates
        spec_chunk: spectral reflectivity within this node
        noise_thres: noise threshold hat separated this peak
        prom_thres: prominence threshold in linear units
        root: flag indicating if root node
        parent_lvl: level of the parent node
    """
    def __init__(self, bounds, spec_chunk, noise_thres, prom_thres, root=False, parent_lvl=0):
        self.bounds = bounds
        self.children = []
        self.level = 0 if root else parent_lvl + 1
        self.root = root
        self.threshold = noise_thres
        self.spec = spec_chunk
        # faster to have prominence filter in linear units
        self.prom_filter = prom_thres
        # prominence filter  2dB (Shupe 2004) or even 6 (Williams 2018)
        #print('at node ', bounds, h.lin2z(noise_thres), spec_chunk)

    def add_noise_sep(self, bounds_left, bounds_right, thres, ignore_prom=False):
        """add a nose separated peak/node
        
        Args:
            bounds_left: boundaries of the left peak
            bounds_right: boundaries of the right peak
            thres: threshold that separates the peaks
        """
        fitting_child = list(filter(lambda x: x.bounds[0] <= bounds_left[0] and x.bounds[1] >= bounds_right[1], self.children))
        if len(fitting_child) == 1:
            #recurse on
            fitting_child[0].add_noise_sep(bounds_left, bounds_right, thres)
        else:
            # insert here
            spec_left = self.spec[bounds_left[0]-self.bounds[0]:bounds_left[1]+1-self.bounds[0]]
            spec_right = self.spec[bounds_right[0]-self.bounds[0]:bounds_right[1]+1-self.bounds[0]]
            prom_left = spec_left[np.nanargmax(spec_left)]/thres
            prom_right = spec_right[np.nanargmax(spec_right)]/thres

            cond_prom = [prom_left > self.prom_filter, prom_right > self.prom_filter]
            if all(cond_prom) or ignore_prom:
                self.children.append(Node(bounds_left, spec_left, thres, self.prom_filter, parent_lvl=self.level))
                self.children.append(Node(bounds_right, spec_right, thres, self.prom_filter, self.prom_filter, parent_lvl=self.level))
            else:
                #print('omitted noise sep. peak at ', bounds_left, bounds_right, h.lin2z(prom_left), h.lin2z(prom_right))
                pass

    #@profile
    def add_min(self, new_index, current_thres, ignore_prom=False):
        """add a local minimum

        Args:
            new_index: bin index of minimum
            current_threshold: reflectivity that separates the peaks
            ignore_prom (optional): ignore the prominence threshold
        """
        if new_index < self.bounds[0] or new_index > self.bounds[1]:
            raise ValueError("child out of parents bounds")
        # this can be simplified for binary trees
        #fitting_child = list(filter(lambda x: x.bounds[0] <= new_index and x.bounds[1] >= new_index, self.children))
        #if len(fitting_child) == 1:
        #    fitting_child[0].add_min(new_index, current_thres)

        if len(self.children) > 0 and self.children[0].bounds[0] <= new_index and self.children[0].bounds[1] >= new_index:
            # append to left child
            self.children[0].add_min(new_index, current_thres)
        elif  len(self.children) > 0 and self.children[1].bounds[0] <= new_index and self.children[1].bounds[1] >= new_index:
            # append to right child
            self.children[1].add_min(new_index, current_thres)
        # or insert here
        else:
            spec_left = self.spec[:new_index+1-self.bounds[0]]
            prom_left = spec_left[np.nanargmax(spec_left)]/current_thres
            # print('spec_chunk left ', self.bounds[0], new_index, h.lin2z(prom_left), spec_left)
            spec_right = self.spec[new_index-self.bounds[0]:]
            prom_right = spec_right[np.nanargmax(spec_right)]/current_thres
            # print('spec_chunk right ', new_index, self.bounds[1], h.lin2z(prom_right), spec_right)

            cond_prom = [prom_left > self.prom_filter, prom_right > self.prom_filter]
            if all(cond_prom) or ignore_prom:
                self.children.append(Node((self.bounds[0], new_index), 
                                     spec_left, current_thres, self.prom_filter, parent_lvl=self.level))
                self.children.append(Node((new_index, self.bounds[1]), 
                                     spec_right, current_thres, self.prom_filter, parent_lvl=self.level))
            #else:
            #    #print('omitted peak at ', new_index, 'between ', self.bounds, h.lin2z(prom_left), h.lin2z(prom_right))
            #    pass 

    def __str__(self):
        string = str(self.level) + ' ' + self.level*'  ' + str(self.bounds) + "   [{:4.1f}]".format(h.lin2z(self.threshold))
        return "{}\n{}".format(string, ''.join([t.__str__() for t in self.children]))


def traverse(Node, coords):
    """traverse a node and recursively all subnodes
    
    Args:
        Node (:class:`Node`): Node object to traverse
        coords: Nodes coordinate as list
    Yields:
        all child nodes recursively"
    """
    #yield {'coords': coords, 'bounds': Node.bounds, 'thres': Node.threshold}
    yield {'coords': coords, 'bounds_left': Node.bounds[0], 'bounds_right': Node.bounds[1], 'thres': Node.threshold}
    for i, n in enumerate(Node.children):
        yield from traverse(n, coords + [i])


def full_tree_id(coord):
    '''convert a coordinate to the id from the full binary tree

    Args:
        coord: Nodes coordinate as a list
    Returns:
        index as in full binary tree
    Example:

        .. code-block:: python

            [0] -> 0
            [0, 1] -> 2
            [0, 0, 0] -> 3
            [0, 1, 1, 0] -> 13
    '''
    idx = 2**(len(coord)-1)-1
    for ind, flag in enumerate(reversed(coord)):
        if flag == 1:
            idx += (2**ind)
    #print(coord,'->',idx)
    return idx


#@profile
def coords_to_id(traversed):
    """calculate the id in level-order from the coordinates

    Args:
        input: traversed tree as list of dict
    Returns:
        traversed tree (dict) with id as key
    """
    traversed_id = {}  
    #print('coords to id, traversed ', traversed) 
    for node in traversed:
        k = full_tree_id(node['coords'])
        traversed_id[k] = node
        parent = [k for k, val in traversed_id.items() if val['coords'] == node['coords'][:-1]]
        traversed_id[k]['parent_id'] = parent[0] if len(parent) == 1 else -1
    # level_no = 0
    # while True:
    #     current_level =list(filter(lambda d: len(d['coords']) == level_no+1, traversed))
    #     if len(current_level) == 0:
    #         break
    #     for d in sorted(current_level, key=lambda d: sum(d['coords'])):
    #         k = full_tree_id(d['coords'])
    #         traversed_id[k] = d
    #         parent = [k for k, val in traversed_id.items() if val['coords'] == d['coords'][:-1]]
    #         traversed_id[k]['parent_id'] = parent[0] if len(parent) == 1 else -1
    #     level_no += 1
    #print('coords to id, traversed_id ', traversed_id)
    return traversed_id

# @jit(nopython=True, fastmath=True)
# def moment(x, Z):
#     """mean, rms, skew for a vel, Z part of the spectrum
#     
#     TODO document
#     Args:
#         x: velocity of bin
#         Z: spectral reflectivity
#     Returns:
#         v, width, skew
#     """
#     # probably arr.sum() is faster than np.sum(arr)
#     sumZ = Z.sum() # memory over processing time
#     #print(Z, sumZ)
#     mean = (x*Z).sum()/sumZ
#     x_mean = x-mean # memory over processing time
#     rms = np.sqrt((((x_mean)**2)*Z).sum()/sumZ)
#     skew = ((((x_mean)**3)*Z)/(sumZ*(rms**3))).sum()
#     #return {'v': mean, 'width': rms, 'skew': skew}
#     return sumZ, mean, rms, skew

#@profile
#@jit(fastmath=True)
@jit(forceobj=True)
def calc_moments(spectrum, bounds, thres, no_cut=False):
    """calc the moments following the formulas given by Görsdorf2015 and Maahn2017

    Args:
        spectrum: spectrum dict
        bounds: boundaries (bin no)
        thres: threshold used
    Returns
        moment, spectrum
    """
    mask = spectrum['specZ_mask'][bounds[0]:bounds[1]+1]
    Z = spectrum['specZ'][bounds[0]:bounds[1]+1][~mask].sum()
    #print('incl noise ', h.lin2z(Z))
    Z = Z - spectrum['noise_lvl'][bounds[0]:bounds[1]+1][~mask].sum()
    #print('wo noise   ', h.lin2z(Z))
    # TODO add the masked processing for the moments
    #spec_masked = np.ma.masked_less(spectrum['specZ'], thres, copy=True)
    if not no_cut:
        masked_Z = h.fill_with(spectrum['specZ'][bounds[0]:bounds[1]+1], 
                        np.logical_or(spectrum['specZ'][bounds[0]:bounds[1]+1]<thres, 
                                      mask), 0.0)
    else:
        masked_Z = h.fill_with(spectrum['specZ'], mask, 0.0)
        masked_Z[:bounds[0]] = 0.0
        masked_Z[bounds[1]+1:] = 0.0

    # seemed to have been quite slow (84us per call or 24% of function)
    mom = moment(spectrum['vel'][bounds[0]:bounds[1]+1], masked_Z)
    moments = {'v': mom[0], 'width': mom[1], 'skew': mom[2]}
    
    #spectrum['specZco'] = spectrum['specZ']/(1+spectrum['specLDR'])
    #spectrum['specZcx'] = (spectrum['specLDR']*spectrum['specZ'])/(1+spectrum['specLDR'])

    #ind_max = spectrum['specSNRco'][bounds[0]:bounds[1]+1].argmax()
    ind_max = np.nanargmax(spectrum['specZ'][bounds[0]:bounds[1]+1])
    #print('Z at argmax ', h.lin2z(spectrum["specZ"][bounds[0]:bounds[1]+1][ind_max-2:ind_max+3]))
    #print('Z at argmax ', spectrum["specZ"][bounds[0]:bounds[1]+1][ind_max-2:ind_max+3])

    prominence = spectrum['specZ'][bounds[0]:bounds[1]+1][ind_max]/thres
    prominence_mask = mask[ind_max]
    moments['prominence'] = prominence if not prominence_mask else 1e-99
    #assert np.all(validSNRco.mask == validSNRcx.mask)
    #print('SNRco', h.lin2z(spectrum['validSNRco'][bounds[0]:bounds[1]+1]))
    #print('SNRcx', h.lin2z(spectrum['validSNRcx'][bounds[0]:bounds[1]+1]))
    #print('LDR', h.lin2z(spectrum['validSNRcx'][bounds[0]:bounds[1]+1]/spectrum['validSNRco'][bounds[0]:bounds[1]+1]))
    moments['z'] = Z
    
    # ldr calculation after the debugging session
    specLDRchunk = spectrum["specLDR"][bounds[0]:bounds[1]+1]
    ldrmax = specLDRchunk[ind_max]
    if np.any(specLDRchunk > 0):
        ldrmin = np.nanmin(specLDRchunk[specLDRchunk > 0])
    else:
        ldrmin = np.nan
    # ldrmax is at maximum of co signal, the minimum should not be smaller,
    # (would indicate a peak not a dip) 
    ldrmin = ldrmax if ldrmin < ldrmax else ldrmin
    #print('ldr ', h.lin2z(spectrum["specLDR"][bounds[0]:bounds[1]+1]))
    #print('ldrmax ', h.lin2z(spectrum["specLDR"][bounds[0]:bounds[1]+1][ind_max-1:ind_max+2]), bounds, ind_max)
    #print('Zcx_validcx', spectrum['specZcx_validcx'][bounds[0]:bounds[1]+1], spectrum['specZcx_validcx'][bounds[0]:bounds[1]+1])
    if np.any(~spectrum['trust_ldr_mask'][bounds[0]:bounds[1]+1]):
        #ldr2 = np.nanmean(spectrum['specLDRmasked'][bounds[0]:bounds[1]+1])
        trust_ldr = spectrum['trust_ldr_mask'][bounds[0]:bounds[1]+1]
        Zcx = spectrum['specZcx'][bounds[0]:bounds[1]+1]
        Zco = spectrum['specZ'][bounds[0]:bounds[1]+1]
        ldr2 = np.nansum(Zcx[~trust_ldr])/np.nansum(Zco[~trust_ldr])
        #ldr2 = (spectrum['specZcx_validcx'][bounds[0]:bounds[1]+1]).sum()/(spectrum['specZ_validcx'][bounds[0]:bounds[1]+1]).sum()
    else:
        ldr2 = np.nan

    #decoup = 10**(spectrum['decoupling']/10.)
    #print('ldr ', h.lin2z(ldr), ' ldrmax ', h.lin2z(ldrmax), 'new ldr ', h.lin2z(ldr2))
    #print('ldr without decoup', h.lin2z(ldr-decoup), ' ldrmax ', h.lin2z(ldrmax-decoup), 'new ldr ', h.lin2z(ldr2-decoup))
    # seemed to have been quite slow (79us per call or 22% of function)
    #moments['ldr'] = ldr2 if (np.isfinite(ldr2) and not np.allclose(ldr2, 0.0)) else np.nan
    if np.isfinite(ldr2) and not np.abs(ldr2) < 0.0001:
        moments['ldr'] = ldr2
    else:
        moments['ldr'] = 1e-6
    #moments['ldr'] = ldr2-decoup
    moments['ldrmax'] = ldrmax
    moments['ldrmin'] = ldrmin

    # is the ldr symmetric?
    vel = spectrum['vel'][bounds[0]:bounds[1]+1]
    vel_step = vel[1] - vel[0]
    width_bins = np.floor(moments['width']/vel_step).astype(int)
    mid_bin = np.floor((bounds[1] - bounds[0])/2).astype(int)
    left_bin = mid_bin - width_bins
    right_bin = mid_bin + width_bins
    #print(bounds, width_bins, mid_bin, vel[mid_bin],
    #      left_bin, vel[left_bin], right_bin, vel[right_bin])
    ldr_left = spectrum['specLDRmasked'][bounds[0]:bounds[1]+1][left_bin]
    ldr_right = spectrum['specLDRmasked'][bounds[0]:bounds[1]+1][right_bin]
    #print(h.lin2z(ldr_left), h.lin2z(ldr_right))
    moments['ldrleft'] = ldr_left
    moments['ldrright'] = ldr_right

    #moments['minv'] = spectrum['vel'][bounds[0]]
    #moments['maxv'] = spectrum['vel'][bounds[1]]

    return moments, spectrum


# @jit(fastmath=True, forceobj=True)
# def calc_moments_STSR(spectrum, bounds, thres, no_cut=False):
#     """calc the moments following the formulas given by Görsdorf2015 and Maahn2017
# 
#     Args:
#         spectrum: spectrum dict
#         bounds: boundaries (bin no)
#         thres: threshold used
#     Returns
#         moment, spectrum
#     """
#     mask = spectrum['specZ_mask'][bounds[0]:bounds[1]+1]
#     Z = spectrum['specZ'][bounds[0]:bounds[1]+1][~mask].sum()
#     # TODO add the masked processing for the moments
#     #spec_masked = np.ma.masked_less(spectrum['specZ'], thres, copy=True)
#     if not no_cut:
#         masked_Z = h.fill_with(spectrum['specZ'][bounds[0]:bounds[1]+1], 
#                         np.logical_or(spectrum['specZ'][bounds[0]:bounds[1]+1]<thres, 
#                                     spectrum['specZ_mask'][bounds[0]:bounds[1]+1]), 0.0)
#     else:
#         masked_Z = h.fill_with(spectrum['specZ'], spectrum['specZ_mask'], 0.0)
#         masked_Z[:bounds[0]] = 0.0
#         masked_Z[bounds[1]+1:] = 0.0
# 
#     # seemed to have been quite slow (84us per call or 24% of function)
#     mom = moment(spectrum['vel'][bounds[0]:bounds[1]+1], masked_Z)
#     moments = {'v': mom[0], 'width': mom[1], 'skew': mom[2]}
# 
#     #ind_max = spectrum['specSNRco'][bounds[0]:bounds[1]+1].argmax()
#     ind_max = np.nanargmax(spectrum['specZ'][bounds[0]:bounds[1]+1])
# 
#     prominence = spectrum['specZ'][bounds[0]:bounds[1]+1][ind_max]/thres
#     prominence_mask = spectrum['specZ_mask'][bounds[0]:bounds[1]+1][ind_max]
#     moments['prominence'] = prominence if not prominence_mask else 1e-99
#     #assert np.all(validSNRco.mask == validSNRcx.mask)
#     #print('SNRco', h.lin2z(spectrum['validSNRco'][bounds[0]:bounds[1]+1]))
#     #print('SNRcx', h.lin2z(spectrum['validSNRcx'][bounds[0]:bounds[1]+1]))
#     #print('LDR', h.lin2z(spectrum['validSNRcx'][bounds[0]:bounds[1]+1]/spectrum['validSNRco'][bounds[0]:bounds[1]+1]))
#     moments['z'] = Z
#     
#     # new try for the peak ldr (sum, then relative; not spectrum of relative with sum)
#     
#     # ldr calculation after the debugging session
#     Rhvmax = spectrum["specRhv"][bounds[0]:bounds[1]+1][ind_max]
#     ldrmax = (1-Rhvmax)/(1+Rhvmax)
#     if not np.all(spectrum['specRhv_mask']):
#         #ldr2 = np.nanmean(spectrum['specLDRmasked'][bounds[0]:bounds[1]+1])
#         rhv = np.nanmean(spectrum['specRhvmasked'][bounds[0]:bounds[1]+1])
#         ldr2 = (1-rhv)/(1+rhv)
#         #print('specLDRmasked', spectrum['specLDRmasked'][bounds[0]:bounds[1]+1])
#     else:
#         ldr2 = np.nan
# 
#     decoup = 10**(spectrum['decoupling']/10.)
#     #print('ldr ', h.lin2z(ldr), ' ldrmax ', h.lin2z(ldrmax), 'new ldr ', h.lin2z(ldr2))
#     #print('ldr without decoup', h.lin2z(ldr-decoup), ' ldrmax ', h.lin2z(ldrmax-decoup), 'new ldr ', h.lin2z(ldr2-decoup))
# 
#     #moments['ldr'] = ldr
#     # seemed to have been quite slow (79us per call or 22% of function)
#     #moments['ldr'] = ldr2 if (np.isfinite(ldr2) and not np.allclose(ldr2, 0.0)) else np.nan
#     if np.isfinite(ldr2) and not np.abs(ldr2) < 0.0001:
#         moments['ldr'] = ldr2
#     else:
#         moments['ldr'] = np.nan
#     #moments['ldr'] = ldr2-decoup
#     moments['ldrmax'] = ldrmax
#     #moments['ldrmax'] = np.nan
# 
#     #moments['minv'] = spectrum['vel'][bounds[0]]
#     #moments['maxv'] = spectrum['vel'][bounds[1]]
# 
#     return moments, spectrum



#@jit(fastmath=True)
# def calc_moments_wo_LDR(spectrum, bounds, thres, no_cut=False):
#     """calc the moments following the formulas given by Görsdorf2015 and Maahn2017
# 
#     Args:
#         spectrum: spectrum dict
#         bounds: boundaries (bin no)
#         thres: threshold used
#     Returns
#         moment, spectrum
#     """
#     mask = spectrum['specZ_mask'][bounds[0]:bounds[1]+1]
#     Z = spectrum['specZ'][bounds[0]:bounds[1]+1][~mask].sum()
#     masked_Z = h.fill_with(spectrum['specZ'], spectrum['specZ_mask'], 0.0)
#     if not no_cut:
#         masked_Z = h.fill_with(masked_Z, (masked_Z<thres), 0.0)
#     mom = moment(spectrum['vel'][bounds[0]:bounds[1]+1], masked_Z[bounds[0]:bounds[1]+1])
#     moments = {'v': mom[0], 'width': mom[1], 'skew': mom[2]}
#     
#     #ind_max = spectrum['specSNRco'][bounds[0]:bounds[1]+1].argmax()
#     ind_max = np.nanargmax(spectrum['specZ'][bounds[0]:bounds[1]+1])
# 
#     prominence = spectrum['specZ'][bounds[0]:bounds[1]+1][ind_max]/thres
#     prominence_mask = spectrum['specZ_mask'][bounds[0]:bounds[1]+1][ind_max]
#     moments['prominence'] = prominence if not prominence_mask else 1e-99
#     moments['z'] = Z
#     moments['ldr'] = 0.0
#     moments['ldr'] = 0.0
#     moments['ldrmax'] = 0.0
#     return moments, spectrum


#@profile
# def tree_from_spectrum(spectrum, peak_finding_params):
#     """generate the tree and return a traversed version
# 
#     .. code-block:: python
# 
#         spectrum = {'ts': self.timestamps[it], 'range': self.range[ir], 'vel': self.velocity,
#             'specZ': specZ, 'noise_thres': specZ.min()}
# 
#     Args:
#         spectrum (dict): spectra
#         peak_finding_params (dict): either from config or manually overwritten 
#     Returns:
#         traversed tree
#     """
# 
#     if 'prom_thres' in peak_finding_params:
#         prom_thres = h.z2lin(peak_finding_params['prom_thres'])
#     else:
#         prom_thres = h.z2lin(1.)
# 
#     # for i in range(spectrum['specZ'].shape[0]):
#     #     if not spectrum['specZ'][i] == 0:
#     #         print(i, spectrum['vel'][i], h.lin2z(spectrum['specZ'][i]))
#     masked_Z = h.fill_with(spectrum['specZ'], spectrum['specZ_mask'], 0)
#     peak_ind = detect_peak_simple(masked_Z, spectrum['noise_thres'])
#     log.info(f"noise thres per peak {peak_ind} {h.lin2z(spectrum['noise_thres']):.3f}")
#     # filter all peaks with that are only 1 bin wide
#     peak_ind = list(filter(lambda e: e[1]-e[0] > 0, peak_ind))
#     if peak_ind:
#         # print('peak ind at noise  level', peak_ind)
#         if len(peak_ind) == 0:
#             t = Node(peak_ind[0], spectrum['specZ'][peak_ind[0]:peak_ind[1]+1], 
#                      spectrum['noise_thres'], prom_thres, root=True)
#         else:
#             t = Node((peak_ind[0][0], peak_ind[-1][-1]), spectrum['specZ'][peak_ind[0][0]:peak_ind[-1][-1]+1], 
#                      spectrum['noise_thres'], prom_thres, root=True)
#             for peak_pair in peak_pairs_to_call(peak_ind):
#                 # print('peak pair', peak_pair)
#                 t.add_noise_sep(peak_pair[0], peak_pair[1], spectrum['noise_thres'])
# 
#         # minima only inside main peaks
#         #minima = get_minima(np.ma.masked_less(spectrum['specZ'], spectrum['noise_thres']*1.1))
#         #minima = get_minima(np.ma.masked_less(masked_Z, spectrum['noise_thres']*1.1))
#         minima = get_minima(h.fill_with(masked_Z, masked_Z<spectrum['noise_thres']*1.1, 1e-30))
#         for m in minima:
#             # print('minimum ', m)
#             if m[1]>spectrum['noise_thres']*1.1:
#                 t.add_min(m[0], m[1])
#             else:
#                 #print('detected minima too low', m[1], spectrum['noise_thres']*1.1)
#                 pass
#         #print(t)
#         traversed = coords_to_id(list(traverse(t, [0])))
#         for i in traversed.keys():
#             #if i == 0:
#             #    moments, spectrum =  calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
#             #else:
#             if 'specZcx' in spectrum:
#                 moments, _ = calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
#             #    alt_moms, _ = calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'], no_cut=True)
#             #    print(traversed[i]['bounds'], alt_moms)
#             else:
#                 moments, _ = calc_moments_wo_LDR(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
#                 #alt_moms, _ = calc_moments_wo_LDR(spectrum, traversed[i]['bounds'], traversed[i]['thres'], no_cut=True)
#                 #print(traversed[i]['bounds'], alt_moms)
#             traversed[i].update(moments)
#             #print('traversed tree')
#             #print(i, traversed[i])
#         #print(print_tree.travtree2text(traversed))
# 
#         #new skewness split
#         chop = False
#         if chop:
#             nodes_with_min = list(traversed.keys())
#             all_parents = [n['parent_id'] for n in traversed.values()]
#             leafs = set(traversed.keys()) - set(all_parents)
#             nodes_to_split = [k for k,v in traversed.items() if k in leafs and np.abs(v['skew']) > 0.4]
#             log.debug(f'all_parents {all_parents} leafs {leafs} nodes to split {nodes_to_split}')
#             calc_chop = lambda x: int(0.85*(x[1]-x[0])+x[0])
#             split_at = [calc_chop(traversed[k]['bounds']) for k in nodes_to_split]
#             log.debug(split_at)
#             for s in split_at:
#                 print('add min', s, masked_Z[s])
#                 t.add_min(s, masked_Z[s], ignore_prom=True)
#             log.debug(t)
#             traversed = coords_to_id(list(traverse(t, [0])))
# 
#             for i in traversed.keys():
#                 moments, _ = calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
#                 traversed[i].update(moments)
#                 if i in nodes_with_min:
#                     traversed[i]['chop'] = True
#             log.debug(traversed)
# 
#     else:
#         traversed = {}
#     return traversed

# def tree_from_peako(spectrum, noise_sep, internal):
# 
#     #add root node
#     if noise_sep:
#         t = Node((noise_sep[0][0], noise_sep[-1][-1]), 
#                 spectrum['specZ'][noise_sep[0][0]:noise_sep[-1][-1]+1], 
#                 spectrum['noise_thres'], root=True)
#         for peak_pair in peak_pairs_to_call(noise_sep):
#             # print('peak pair', peak_pair)
#             t.add_noise_sep(peak_pair[0], peak_pair[1], spectrum['noise_thres'])
#         for m in internal:
#             t.add_min(m, spectrum['specZ'][m], ignore_prom=True)
# 
#         log.debug(t)
#         traversed = coords_to_id(list(traverse(t, [0])))
#         for i in traversed.keys():
#             moments, _ = calc_moments_wo_LDR(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
#             traversed[i].update(moments)
#             #print('traversed tree')
#             # if moments['v'] == 0.0 or not np.isfinite(moments['v']):
#             #     print(i, traversed[i])
#             #     input()
# 
#         log.debug(print_tree.travtree2text(traversed))
#     else:
#         traversed = {}
#     return traversed


# # LEGACY to be removed
# #
# def find_edges(spectrum, fill_value, peak_locations):
#     """
#     Find the indices of left and right edges of peaks in a spectrum
# 
#     Args:
#         spectrum: a single spectrum in linear units
#         peak_locations: indices of peaks detected for this spectrum
#         fill_value: The fill value which indicates the spectrum is below noise floor
# 
#     Returns:
#         left_edges: list of indices of left edges,
#         right_edges: list of indices of right edges
#     """
#     left_edges = []
#     right_edges = []
# 
#     for p_ind in range(len(peak_locations)):
#         # start with the left edge
#         p_l = peak_locations[p_ind]
# 
#         # set first estimate of left edge to last bin before the peak
#         closest_below_noise_left = np.where(spectrum[0:p_l] <= fill_value)
#         if len(closest_below_noise_left[0]) == 0:
#             closest_below_noise_left = 0
#         else:
#             # add 1 to get the first bin of the peak which is not fill_value
#             closest_below_noise_left = max(closest_below_noise_left[0]) + 1
# 
#         if p_ind == 0:
#             # if this is the first peak, the left edge is the closest_below_noise_left
#             left_edge = closest_below_noise_left
#         elif peak_locations[p_ind - 1] > closest_below_noise_left:
#             # merged peaks
#             left_edge = np.argmin(spectrum[peak_locations[p_ind - 1]: p_l])
#             left_edge = left_edge + peak_locations[p_ind - 1]
#         else:
#             left_edge = closest_below_noise_left
# 
#         # Repeat for right edge
#         closest_below_noise_right = np.where(spectrum[p_l:-1] <= fill_value)
#         if len(closest_below_noise_right[0]) == 0:
#             # if spectrum does not go below noise (fill value), set it to the last bin
#             closest_below_noise_right = len(spectrum) - 1
#         else:
#             # subtract one to obtain the last index of the peak
#             closest_below_noise_right = min(closest_below_noise_right[0]) + p_l - 1
# 
#         # if this is the last (rightmost) peak, this first guess is the right edge
#         if p_ind == (len(peak_locations) - 1):
#             right_edge = closest_below_noise_right
# 
#         elif peak_locations[p_ind + 1] < closest_below_noise_right:
#             right_edge = np.argmin(spectrum[p_l:peak_locations[p_ind + 1]]) + p_l
#         else:
#             right_edge = closest_below_noise_right
# 
#         left_edges.append(np.int(left_edge))
#         right_edges.append(np.int(right_edge))
# 
#     return left_edges, right_edges


def bounds_from_find_peak(peaks, prop):

    left_bases = prop['left_bases']
    right_bases = prop['right_bases']
    bases = sorted(list(set(left_bases.tolist() + right_bases.tolist())))
    #print(peaks, bases)

    bounds = []
    for ip in peaks:
        il = np.searchsorted(bases, ip)
        bounds.append([bases[il-1], bases[il]])

    return bounds


def remove_gap_peaks(locs, props, gaps):
    """remove peaks detected where gaps were filled with raw spectrum values
    """
    index = [i for i in range(len(locs)) if not locs[i] in gaps]
    locs = locs[index]
    props_out = {}
    for key, val in props.items():
        props_out[key] = val[index]
    return locs, props_out


def fix_peaks_unique(peaks, prop):
    """equally high peaks are not prominence filtered by find_peaks
    (actually specified behavior)
    
    i.e. scipy.signal.find_peaks(np.array([0,0,1,4,5,1,5,3,0]), prominence=1)
    >> (array([4, 6]),
        {'prominences': array([5., 5.]),
         'left_bases': array([1, 1]),
         'right_bases': array([8, 8])})
    
    """

    log.warning("temporary fix for the prominence of equally high peaks")
    d = {}
    for e in zip(prop['left_bases'], prop['right_bases'], peaks):
        d[e[:2]] = e[2]

    peaks = d.values()
    lr_bounds = d.keys()
    prop['left_bases'] = np.array([e[0] for e in lr_bounds])
    prop['right_bases'] = np.array([e[1] for e in lr_bounds])

    return peaks, prop


# def tree_from_spectrum_peako(spectrum, peak_finding_params, gaps=None):
#     """generate the tree and return a traversed version use peako-like peakfinder
# 
#     Args:
#         spectrum (dict): spectra
#         peak_finding_params (dict): either from config or manually overwritten 
#     Returns:
#         traversed tree
#     """
# 
#     #print('noise ', spectrum['noise_thres'], h.lin2z(spectrum['noise_thres']))
#     # scipy.signal.find_peaks cannot deal with nans, i.e. lin2z([... 0 ... ]) causes problems
#     masked_Z_pf = h.fill_with(spectrum['specZ'], (spectrum['specZ_mask'] | (spectrum['specZ'] < spectrum['noise_thres'])), spectrum['noise_thres']/4.)
#     if 'specSNRco_mask' in spectrum:
#         masked_Z_pf = h.fill_with(masked_Z_pf, spectrum['specSNRco_mask'], spectrum['noise_thres']/4.)
#     #print('masked_Z_p', h.lin2z(masked_Z_pf).tolist())
#     masked_Z = h.fill_with(spectrum['specZ'], (spectrum['specZ_mask']), 0)
# 
#     width = peak_finding_params['width_thres']/peak_finding_params['vel_step']
#     locs, props = scipy.signal.find_peaks(
#         h.lin2z(masked_Z_pf), 
#         height=h.lin2z(spectrum['noise_thres']),
#         prominence=peak_finding_params['prom_thres'],
#         width=width,
#         rel_height=0.5)
#     if gaps is not None:
#         locs, props = remove_gap_peaks(locs, props, np.where(gaps)[0])
#     log.debug(f'find_peaks locs {locs} props {props}')
#     #noise_floor_edges = detect_peak_simple(masked_Z_pf, spectrum['noise_thres'])
#     #print('noise_floor_edges', noise_floor_edges)
# 
#     if np.any(np.unique(h.lin2z(masked_Z_pf)[locs], return_counts=True)[1] > 1):
#         locs, props = fix_peaks_unique(locs, props)
#     bounds = bounds_from_find_peak(locs, props)
#     #le, re = find_edges(h.lin2z(masked_Z_pf), h.lin2z(spectrum['noise_thres']), locs)
#     # when locs are sorted, this cuts the rightmost peaks
#     #locs = locs[0: max_peaks] if len(locs) > max_peaks else locs
#     #print('final locs ', locs, list(zip(le, re)))
#     #bounds = list(zip(le, re))
# 
#     # and now the peaktree part
#     # take ideas from the first implementation
#     if not all([e[0]<e[1] for e in bounds]):
#         bounds = []
#     noise_sep, internal = h.divide_bounds(bounds)
#     log.info(f"sep internal {bounds} => {noise_sep} {internal}")
# 
#     # the internal peaks have to be sorted by their height
#     # otherwise the tree will not be build correctly
#     internal = np.array(internal)[np.argsort(masked_Z[internal])]
# 
#     if noise_sep:
#         t = Node((noise_sep[0][0], noise_sep[-1][-1]), 
#                 spectrum['specZ'][noise_sep[0][0]:noise_sep[-1][-1]+1], 
#                 spectrum['noise_thres'], peak_finding_params['prom_thres'], root=True)
#         for peak_pair in peak_pairs_to_call(noise_sep):
#             t.add_noise_sep(peak_pair[0], peak_pair[1], spectrum['noise_thres'],
#                             ignore_prom=True)
#         for m in internal:
#             t.add_min(m, spectrum['specZ'][m], ignore_prom=True)
# 
#         #print(t)
#         traversed = coords_to_id(list(traverse(t, [0])))
#         for i in traversed.keys():
#             #print(i, traversed[i]['bounds'], h.lin2z(traversed[i]['thres']))
#             #moments, _ = calc_moments_wo_LDR(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
#             if spectrum['polarimetry'] == 'LDR':
#                 moments, _ = calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
#             elif spectrum['polarimetry'] == 'STSR':
#                 moments, _ = calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
#                 # with the new SLDR method one could revert to the LDR moment calc
#                 #moments, _ = calc_moments_STSR(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
#             elif spectrum['polarimetry'] == 'false':
#                 moments, _ = calc_moments_wo_LDR(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
# 
#             if 'cal_offset' in peak_finding_params:
#                 moments['z'] *= h.z2lin(peak_finding_params['cal_offset'])
#             traversed[i].update(moments)
#             #print('traversed tree')
#             # if moments['v'] == 0.0 or not np.isfinite(moments['v']):
#             #     print(i, traversed[i])
#             #     input()
#             #if spectrum['polarimetry'] in ['LDR', 'STSR']:
#             #    print(f"{traversed[i]['bounds']} ldrs {h.lin2z(moments['ldrmin']):.2f}  ", 
#             #          f"{h.lin2z(moments['ldrleft']):.2f} {h.lin2z(moments['ldrright']):.2f}")
# 
#         log.debug(print_tree.travtree2text(traversed))
# 
#     else:
#         traversed = {}
#     return traversed


def spectrum_to_tree(vel_step, spectrum, mask, params):
    """convert spectrum to tree
    
    
    Parameters
    ----------
    vel_step : float
        velocity spacing of spectrum
    spectrum : array
        Reflectivity (linear units?)
    mask : array
        Mask indicating valid measurements (i.e. above noise)
    params : dict
        peak finding parameters

    Returns
    -------
    
    Notes
    -----
    
    Replaces old tree_from_spectrum{_peako} function with a more modularized interface,
    i.a. use plain numpy arrays as input instead of dict

    TODO: METEK Bauer Blur not considered for the moment

    """

    if mask.all():
        return {}
    
    spectrum_min = spectrum[~mask].min()
    # scipy.signal.find_peaks cannot deal with nans, i.e. lin2z([... 0 ... ]) causes problems
    masked_spectrum = h.fill_with(spectrum, mask, spectrum_min/2)

    width = params['width_thres']/vel_step
    locs, props = scipy.signal.find_peaks(
        h.lin2z(masked_spectrum), 
        height=h.lin2z(spectrum_min),
        prominence=params['prom_thres'],
        width=width,
        rel_height=0.5)
    log.debug(f'find_peaks locs {locs} props {props}')
    
    if np.any(np.unique(h.lin2z(masked_spectrum)[locs], return_counts=True)[1] > 1):
        locs, props = fix_peaks_unique(locs, props)
    bounds = bounds_from_find_peak(locs, props)
    
    # and now the peaktree part
    if not all([e[0]<e[1] for e in bounds]):
        bounds = []
    noise_sep, internal = h.divide_bounds(bounds)
    log.info(f"sep internal {bounds} => {noise_sep} {internal}")

    # the internal peaks have to be sorted by their height
    # otherwise the tree will not be build correctly
    internal = np.array(internal)[np.argsort(masked_spectrum[internal])]

    if noise_sep:
        t = Node((noise_sep[0][0], noise_sep[-1][-1]), 
                spectrum[noise_sep[0][0]:noise_sep[-1][-1]+1], 
                spectrum_min, params['prom_thres'], root=True)
        for peak_pair in peak_pairs_to_call(noise_sep):
            t.add_noise_sep(peak_pair[0], peak_pair[1], spectrum_min,
                            ignore_prom=True)
        for m in internal:
            t.add_min(m, spectrum[m], ignore_prom=True)

        traversed = coords_to_id(list(traverse(t, [0])))
    else:
        traversed = {}

    return traversed
    

def moments(node, vel, spectrum, moms=['M0', 'M1', 'M2', 'M3', 'P']):
    """Compute selected spectral moments and a peak-to-threshold ratio within a bounded velocity range.

    Extracts a sub-spectrum defined by node['bounds_left'] and node['bounds_right'] (inclusive).
    For M1-M3, values below node['thres'] are zeroed before intensity-weighted computations.

    Parameters
    ----------
    node : dict
        single node of a tree, must at least contain:
        - 'bounds_left' (int): Inclusive left index.
        - 'bounds_right' (int): Inclusive right index.
        - 'thres' (float): Threshold for M1-M3 and denominator for P.
    vel : array_like, shape (N,)
        Velocity array aligned with spectrum.
    spectrum : array_like, shape (N,)
        Spectral power/reflectivity aligned with vel.
    moms : sequence of {'M0', 'M1', 'M2', 'M3', 'P'}, optional
        Moments/metrics to compute. Default is all.

    Returns
    -------
    dict
        Mapping from requested moment names to values:
        - M0: Sum of power in bounds.
        - M1: Intensity-weighted mean velocity (thresholded).
        - M2: Intensity-weighted standard deviation (thresholded).
        - M3: Intensity-weighted skewness (thresholded).
        - P : Peak(Z_chunk) / thres.

    Notes
    -----
    - Thresholding uses h.fill_with(Z_chunk, Z_chunk < thres, 0).
    - M2 and M3 require M1; include 'M1' in moms if requesting 'M2' or 'M3'.
    - Divisions by zero can occur if sum(Z_thres) == 0 (M1–M3), M2 == 0 (M3), or thres == 0 (P).
    - P uses numpy.nanargmax; all-NaN windows will raise a ValueError.
    """

    mom = {}
        
    bounds_l = node['bounds_left']
    bounds_r = node['bounds_right']
    thres = node['thres']
    Z_chunk = spectrum[bounds_l:bounds_r+1]
    if 'M0' in moms:
        mom['M0'] = Z_chunk.sum()
    if 'M1' in moms:
        Z_chunk_thres = h.fill_with(Z_chunk, Z_chunk < thres, 0)
        vel_chunk = vel[bounds_l:bounds_r+1]
        M0_thres = Z_chunk_thres.sum()
        mom['M1'] = (vel_chunk*Z_chunk_thres).sum() / M0_thres
        vel_m_M1 = vel_chunk - mom['M1']
    if 'M2' in moms:
        mom['M2']  = np.sqrt((vel_m_M1**2 * Z_chunk_thres).sum() / M0_thres)
    if 'M3' in moms:
        mom['M3'] = ((vel_m_M1**3 * Z_chunk_thres).sum() / (M0_thres * mom['M2']**3 ))

    if 'P' in moms:
        ind_max = np.nanargmax(Z_chunk)
        mom['P'] = Z_chunk[ind_max] / thres

    return mom



def add_moments(tree, vel, variables):
    """Compute spectral moments for multiple variables at each node and attach results to a copied tree.

    For each node in `tree`, computes moments over the node's bounded window for every
    spectrum in `variables` using `moments(node, vel, spectrum)` (default set of moments),
    and stores them under node['moments'][var_name].
    
    Parameters
    ----------
    tree : Mapping
        Dictionary of nodes. Each node must contain at least:
        - 'bounds_left' (int), 'bounds_right' (int): Inclusive indices into `vel`/spectra.
        - 'thres' (float): Threshold used by `moments`.
    vel : array_like, shape (N,)
        1D velocity array aligned with all spectra in `variables`.
    variables : Mapping[str, array_like]
        Mapping from variable name (e.g., 'Z', 'Zcx', 'LDR') to a 1D spectrum array
        aligned with `vel`.
    
    Returns
    -------
    dict
        Deep copy of `tree` where each node has an added key 'moments', a dict mapping
        variable names to the dict returned by `moments`.
    
    Notes
    -----
    - Uses a deep copy; original `tree` is not modified.
    - Errors from `moments` (e.g., out-of-bounds indices, zero divisions, all-NaN peaks)
      may propagate.
    """
    
    tree_result = copy.deepcopy(tree)
    for i, node in tree.items():
        tree_result[i]['moments'] = {}
        for k, var in variables.items():
            tree_result[i]['moments'][k] = moments(
                node, vel, var)
    
    return tree_result
    


def tree_to_ds(tree, max_n_nodes=15):
    """ """
    
    ds = xr.Dataset(coords={'nodes': np.array(list(tree.keys()))})

    ds['bounds_left'] = ('nodes', [n['bounds_left'] for n in tree.values()])
    ds['bounds_right'] = ('nodes', [n['bounds_right'] for n in tree.values()])
    ds['thres'] = ('nodes', [n['thres'] for n in tree.values()])
    ds['parent_id'] = ('nodes', [n['parent_id'] for n in tree.values()])

    for k, var in tree[0]['moments'].items():
        for m in var:
            ds[f'{k}_{m}'] = ('nodes', [n['moments'][k][m] for n in tree.values()])

    ds = ds.reindex(nodes=np.arange(max_n_nodes), fill_value=-999)
    return ds


def tree_to_numpy(tree, max_n_nodes=15):
    """ """
    
    # TODO separate out float and int

    variables = ['bounds_left', 'bounds_right', 'parent_id', 'thres']

    output = np.zeros((len(variables), max_n_nodes))
    indices = np.array(list(tree.keys()))
    indices = indices[indices < max_n_nodes]
    #print(indices)
    for i, var in enumerate(variables):
        if len(indices) > 0:
            output[i,indices] =  [tree[j][var] for j in indices]

    #print(output.shape, len(variables))

    return output, np.array(variables)

        
    
    ds = xr.Dataset(coords={'nodes': np.array(list(tree.keys()))})

    ds['bounds_left'] = ('nodes', [n['bounds'][0] for n in tree.values()])
    ds['bounds_right'] = ('nodes', [n['bounds'][1] for n in tree.values()])
    ds['parent_id'] = ('nodes', [n['parent_id'] for n in tree.values()])
    ds['thres'] = ('nodes', [n['thres'] for n in tree.values()])

    for k, var in tree[0]['moments'].items():
        for m in var:
            ds[f'{k}_{m}'] = ('nodes', [n['moments'][k][m] for n in tree.values()])

    ds = ds.reindex(nodes=np.arange(max_n_nodes), fill_value=-999)
    return ds

    


def ufunc_wrapper(vel, variables, mask, var_peak_finding=None, vel_step=None, params=None):
    """ """
    

    tree = spectrum_to_tree(
        vel_step, variables[var_peak_finding], mask, params
    )

    tree = add_moments(
        tree, vel, variables
    )

    return tree_to_numpy(tree)
    
def ufunc_wrapper_Zonly(vel, Z, mask, vel_step=None, params=None):
    """ """
    

    #print('wrapper start')
    tree = spectrum_to_tree(
        vel_step, Z, mask, params
    )
    #print(tree)

    tree = add_moments(
        tree, vel, {'Z': Z}
    )
    #print(tree)

    return tree_to_numpy(tree)