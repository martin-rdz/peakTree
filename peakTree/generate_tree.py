""""""
"""
Author: radenz@tropos.de
"""

import matplotlib
matplotlib.use('Agg')

import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from . import helpers as h
from . import print_tree
import toml
from operator import itemgetter
from numba import jit

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


@jit(fastmath=True)
def split_peak_ind_by_space(peak_ind):
    """split a list of peak indices by their maximum space
    use for noise floor separated peaks
    
    Args:
        peak_ind: list of peak indices ``[(163, 165), (191, 210), (222, 229), (248, 256)]``
    Returns:
        left sublist, right sublist
    """
    if len(peak_ind) == 1:
        return [peak_ind, peak_ind]
    left_i = np.array([elem[0] for elem in peak_ind])
    right_i = np.array([elem[1] for elem in peak_ind])
    spacing = left_i[1:]-right_i[:-1]
    split_i = np.argmax(spacing)
    return peak_ind[:split_i+1], peak_ind[split_i+1:]


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

    def add_noise_sep(self, bounds_left, bounds_right, thres):
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
            prom_left = spec_left[spec_left.argmax()]/thres
            prom_right = spec_right[spec_right.argmax()]/thres
            if prom_left > self.prom_filter and prom_right > self.prom_filter:
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
            prom_left = spec_left[spec_left.argmax()]/current_thres
            # print('spec_chunk left ', self.bounds[0], new_index, h.lin2z(prom_left), spec_left)
            spec_right = self.spec[new_index-self.bounds[0]:]
            prom_right = spec_right[spec_right.argmax()]/current_thres
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
        string = str(self.level) + ' ' + self.level*'  ' + str(self.bounds) + "   [{:4.3e}]".format(self.threshold)
        return "{}\n{}".format(string, ''.join([t.__str__() for t in self.children]))


def traverse(Node, coords):
    """traverse a node and recursively all subnodes
    
    Args:
        Node (:class:`Node`): Node object to traverse
        coords: Nodes coordinate as list
    Yields:
        all child nodes recursively"
    """
    yield {'coords': coords, 'bounds': Node.bounds, 'thres': Node.threshold}
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

@jit(nopython=True, fastmath=True)
def moment(x, Z):
    """mean, rms, skew for a vel, Z part of the spectrum
    
    Args:
        x: velocity of bin
        Z: spectral reflectivity
    Returns:
        dict with v, width, skew
    """
    # probably arr.sum() is faster than np.sum(arr)
    sumZ = Z.sum() # memory over processing time
    mean = (x*Z).sum()/sumZ
    x_mean = x-mean # memory over processing time
    rms = np.sqrt((((x_mean)**2)*Z).sum()/sumZ)
    skew = ((((x_mean)**3)*Z)/(sumZ*(rms**3))).sum()
    #return {'v': mean, 'width': rms, 'skew': skew}
    return mean, rms, skew

#@profile
@jit(fastmath=True)
def calc_moments(spectrum, bounds, thres, no_cut=False):
    """calc the moments following the formulas given by Görsdorf2015 and Maahn2017

    Args:
        spectrum: spectrum dict
        bounds: boundaries (bin no)
        thres: threshold used
    Returns
        moment, spectrum
    """
    Z = spectrum['specZ'][bounds[0]:bounds[1]+1].sum()
    # TODO add the masked pocessing for the moments
    #spec_masked = np.ma.masked_less(spectrum['specZ'], thres, copy=True)
    if not no_cut:
        masked_Z = h.fill_with(spectrum['specZ'][bounds[0]:bounds[1]+1], 
                        np.logical_or(spectrum['specZ'][bounds[0]:bounds[1]+1]<thres, 
                                    spectrum['specZ_mask'][bounds[0]:bounds[1]+1]), 0.0)
    else:
        masked_Z = h.fill_with(spectrum['specZ'], spectrum['specZ_mask'], 0.0)
        masked_Z[:bounds[0]] = 0.0
        masked_Z[bounds[1]+1:] = 0.0

    # seemed to have been quite slow (84us per call or 24% of function)
    mom = moment(spectrum['vel'][bounds[0]:bounds[1]+1], masked_Z)
    moments = {'v': mom[0], 'width': mom[1], 'skew': mom[2]}
    
    #spectrum['specZco'] = spectrum['specZ']/(1+spectrum['specLDR'])
    #spectrum['specZcx'] = (spectrum['specLDR']*spectrum['specZ'])/(1+spectrum['specLDR'])

    #ind_max = spectrum['specSNRco'][bounds[0]:bounds[1]+1].argmax()
    ind_max = spectrum['specZ'][bounds[0]:bounds[1]+1].argmax()

    prominence = spectrum['specZ'][bounds[0]:bounds[1]+1][ind_max]/thres
    prominence_mask = spectrum['specZ_mask'][bounds[0]:bounds[1]+1][ind_max]
    moments['prominence'] = prominence if not prominence_mask else 1e-99
    #assert np.all(validSNRco.mask == validSNRcx.mask)
    #print('SNRco', h.lin2z(spectrum['validSNRco'][bounds[0]:bounds[1]+1]))
    #print('SNRcx', h.lin2z(spectrum['validSNRcx'][bounds[0]:bounds[1]+1]))
    #print('LDR', h.lin2z(spectrum['validSNRcx'][bounds[0]:bounds[1]+1]/spectrum['validSNRco'][bounds[0]:bounds[1]+1]))
    moments['z'] = Z
    
    # new try for the peak ldr (sum, then relative; not spectrum of relative with sum)
    #
    # omit this section, it only takes time
    #
    # if not np.all(spectrum['specZ_mask']) and not np.all(spectrum['specZcx_mask']):
    #     min_Zco = np.min(spectrum["specZ"][~spectrum['specZ_mask']])
    #     min_Zcx = np.min(spectrum["specZcx"][~spectrum['specZcx_mask']])
    #     Zco = spectrum['specZ']
    #     Zcx = spectrum['specZcx']
    #     #ldr = np.nansum(masked_Zcx[bounds[0]:bounds[1]+1]/min_Zco)/np.sum(masked_Zco[bounds[0]:bounds[1]+1]/min_Zcx)
    #     ldr = np.nansum(Zcx[bounds[0]:bounds[1]+1])/(Zco[bounds[0]:bounds[1]+1]).sum()
    #     ldrmax = spectrum["specLDR"][bounds[0]:bounds[1]+1][ind_max]
    # else:
    #     ldr = np.nan
    #     ldrmax = np.nan
    
    # ldr calculation after the debugging session
    ldrmax = spectrum["specLDR"][bounds[0]:bounds[1]+1][ind_max]
    if not np.all(spectrum['specZcx_mask']):
        ldr2 = (spectrum['specZcx_validcx'][bounds[0]:bounds[1]+1]).sum()/(spectrum['specZ_validcx'][bounds[0]:bounds[1]+1]).sum()
    else:
        ldr2 = np.nan

    decoup = 10**(spectrum['decoupling']/10.)
    #print('ldr ', h.lin2z(ldr), ' ldrmax ', h.lin2z(ldrmax), 'new ldr ', h.lin2z(ldr2))
    #print('ldr without decoup', h.lin2z(ldr-decoup), ' ldrmax ', h.lin2z(ldrmax-decoup), 'new ldr ', h.lin2z(ldr2-decoup))

    # for i in range(spectrum['specZcx_validcx'].shape[0]):
    #     print(i, h.lin2z(np.array([spectrum['specZ'][i], spectrum['specZcx_validcx'][i], spectrum['specZ_validcx'][i]])), spectrum['specZcx_mask'][i])

    #moments['ldr'] = ldr
    # seemed to have been quite slow (79us per call or 22% of function)
    #moments['ldr'] = ldr2 if (np.isfinite(ldr2) and not np.allclose(ldr2, 0.0)) else np.nan
    if np.isfinite(ldr2) and not np.abs(ldr2) < 0.0001:
        moments['ldr'] = ldr2
    else:
        moments['ldr'] = np.nan
    #moments['ldr'] = ldr2-decoup
    moments['ldrmax'] = ldrmax

    #moments['minv'] = spectrum['vel'][bounds[0]]
    #moments['maxv'] = spectrum['vel'][bounds[1]]

    return moments, spectrum


@jit(fastmath=True)
def calc_moments_wo_LDR(spectrum, bounds, thres, no_cut=False):
    """calc the moments following the formulas given by Görsdorf2015 and Maahn2017

    Args:
        spectrum: spectrum dict
        bounds: boundaries (bin no)
        thres: threshold used
    Returns
        moment, spectrum
    """
    Z = (spectrum['specZ'][bounds[0]:bounds[1]+1]).sum()
    masked_Z = h.fill_with(spectrum['specZ'], spectrum['specZ_mask'], 0.0)
    if not no_cut:
        masked_Z = h.fill_with(masked_Z, (masked_Z<thres), 0.0)
    mom = moment(spectrum['vel'][bounds[0]:bounds[1]+1], masked_Z[bounds[0]:bounds[1]+1])
    moments = {'v': mom[0], 'width': mom[1], 'skew': mom[2]}
    
    #ind_max = spectrum['specSNRco'][bounds[0]:bounds[1]+1].argmax()
    ind_max = spectrum['specZ'][bounds[0]:bounds[1]+1].argmax()

    prominence = spectrum['specZ'][bounds[0]:bounds[1]+1][ind_max]/thres
    prominence_mask = spectrum['specZ_mask'][bounds[0]:bounds[1]+1][ind_max]
    moments['prominence'] = prominence if not prominence_mask else 1e-99
    moments['z'] = Z
    moments['ldr'] = 0.0
    moments['ldr'] = 0.0
    moments['ldrmax'] = 0.0
    return moments, spectrum


#@profile
def tree_from_spectrum(spectrum, peak_finding_params):
    """generate the tree and return a traversed version

    .. code-block:: python

        spectrum = {'ts': self.timestamps[it], 'range': self.range[ir], 'vel': self.velocity,
            'specZ': specZ, 'noise_thres': specZ.min()}

    Args:
        spectrum (dict): spectra dict
    Returns:
        traversed tree
    """

    if 'prom_thres' in peak_finding_params:
        prom_thres = h.z2lin(peak_finding_params['prom_thres'])
    else:
        prom_thres = h.z2lin(1.)

    # for i in range(spectrum['specZ'].shape[0]):
    #     if not spectrum['specZ'][i] == 0:
    #         print(i, spectrum['vel'][i], h.lin2z(spectrum['specZ'][i]))
    masked_Z = h.fill_with(spectrum['specZ'], spectrum['specZ_mask'], 0)
    peak_ind = detect_peak_simple(masked_Z, spectrum['noise_thres'])
    # filter all peaks with that are only 1 bin wide
    peak_ind = list(filter(lambda e: e[1]-e[0] > 0, peak_ind))
    if peak_ind:
        # print('peak ind at noise  level', peak_ind)
        if len(peak_ind) == 0:
            t = Node(peak_ind[0], spectrum['specZ'][peak_ind[0]:peak_ind[1]+1], 
                     spectrum['noise_thres'], prom_thres, root=True)
        else:
            t = Node((peak_ind[0][0], peak_ind[-1][-1]), spectrum['specZ'][peak_ind[0][0]:peak_ind[-1][-1]+1], 
                     spectrum['noise_thres'], prom_thres, root=True)
            for peak_pair in peak_pairs_to_call(peak_ind):
                # print('peak pair', peak_pair)
                t.add_noise_sep(peak_pair[0], peak_pair[1], spectrum['noise_thres'])

        # minima only inside main peaks
        #minima = get_minima(np.ma.masked_less(spectrum['specZ'], spectrum['noise_thres']*1.1))
        #minima = get_minima(np.ma.masked_less(masked_Z, spectrum['noise_thres']*1.1))
        minima = get_minima(h.fill_with(masked_Z, masked_Z<spectrum['noise_thres']*1.1, 1e-30))
        for m in minima:
            # print('minimum ', m)
            if m[1]>spectrum['noise_thres']*1.1:
                t.add_min(m[0], m[1])
            else:
                #print('detected minima too low', m[1], spectrum['noise_thres']*1.1)
                pass
        #print(t)
        traversed = coords_to_id(list(traverse(t, [0])))
        for i in traversed.keys():
            #if i == 0:
            #    moments, spectrum =  calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
            #else:
            if 'specZcx' in spectrum:
                moments, _ = calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
            #    alt_moms, _ = calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'], no_cut=True)
            #    print(traversed[i]['bounds'], alt_moms)
            else:
                moments, _ = calc_moments_wo_LDR(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
                #alt_moms, _ = calc_moments_wo_LDR(spectrum, traversed[i]['bounds'], traversed[i]['thres'], no_cut=True)
                #print(traversed[i]['bounds'], alt_moms)
            traversed[i].update(moments)
            #print('traversed tree')
            #print(i, traversed[i])
        #print(print_tree.travtree2text(traversed))

        #new skewness split
        chop = False
        if chop:
            nodes_with_min = list(traversed.keys())
            all_parents = [n['parent_id'] for n in traversed.values()]
            leafs = set(traversed.keys()) - set(all_parents)
            nodes_to_split = [k for k,v in traversed.items() if k in leafs and np.abs(v['skew']) > 0.4]
            print('all_parents ', all_parents, 'leafs ', leafs, ' nodes to split ', nodes_to_split)
            calc_chop = lambda x: int(0.85*(x[1]-x[0])+x[0])
            split_at = [calc_chop(traversed[k]['bounds']) for k in nodes_to_split]
            print(split_at)
            for s in split_at:
                print('add min', s, masked_Z[s])
                t.add_min(s, masked_Z[s], ignore_prom=True)
            print(t)
            traversed = coords_to_id(list(traverse(t, [0])))

            for i in traversed.keys():
                moments, _ = calc_moments(spectrum, traversed[i]['bounds'], traversed[i]['thres'])
                traversed[i].update(moments)
                if i in nodes_with_min:
                    traversed[i]['chop'] = True
            print(traversed)

    else:
        traversed = {}
    return traversed
