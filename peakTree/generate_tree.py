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
        traversed_id[k]['id_parent'] = parent[0] if len(parent) == 1 else -1
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
    

def moments(bounds_l, bounds_r, thres, vel, spectrum, moms=['M0', 'M1', 'M2', 'M3', 'P']):
    """Compute selected spectral moments and a peak-to-threshold ratio within a bounded velocity range.

    Extracts a sub-spectrum defined by bounds_left and bounds_right (inclusive).
    For M1-M3, values below thres are zeroed before intensity-weighted computations.

    Parameters
    ----------
    bounds_l : int
        Inclusive left index
    bounds_r : int
        Inclusive right index
    thres : float
        Threshold for M1-M3 and denominator for P
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
    - Divisions by zero can occur if sum(Z_thres) == 0 (M1-M3), M2 == 0 (M3), or thres == 0 (P).
    - P uses numpy.nanargmax; all-NaN windows will raise a ValueError.
    """

    mom = {}
        
    Z_chunk = spectrum[bounds_l:bounds_r+1]
    #print(bounds_l, bounds_r, 10*np.log10(Z_chunk))
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



def add_moments(tree, vel, a, inputvarnames, meta):
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
    vel : array_like, shape (nfft,)
        1D velocity array aligned with all spectra in `variables`.
    a : array_like, shape (no_vars, nfft)
        ...
    inputvarnames : list, shape (no_vars)
        Name of first dimensions of array a (meta keys are referring to that name).
    meta : Mapping
        Definition of what Moments should be calculated for which variable
    
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
        for k, var in meta.items():
            j_in_array = np.argwhere(inputvarnames == k)[0][0]
            #print(k, var, j_in_array, a.shape)
            tree_result[i]['moments'][k] = moments(
                node['bounds_left'],
                node['bounds_right'],
                node['thres'],
                vel,
                a[j_in_array,:],
                moms=var
                )
    
    return tree_result
    


def tree_to_ds(tree, max_n_nodes=15):
    """ """
    
    ds = xr.Dataset(coords={'nodes': np.array(list(tree.keys()))})

    ds['bounds_left'] = ('nodes', [n['bounds_left'] for n in tree.values()])
    ds['bounds_right'] = ('nodes', [n['bounds_right'] for n in tree.values()])
    ds['thres'] = ('nodes', [n['thres'] for n in tree.values()])
    ds['id_parent'] = ('nodes', [n['id_parent'] for n in tree.values()])

    for k, var in tree[0]['moments'].items():
        for m in var:
            ds[f'{k}_{m}'] = ('nodes', [n['moments'][k][m] for n in tree.values()])

    ds = ds.reindex(nodes=np.arange(max_n_nodes), fill_value=-999)
    return ds


def tree_to_numpy_dtypeobject(tree):
    """Convert a tree to a np array of dtype object for use with xarrays apply_ufunc.

    Parameters
    ----------
    tree : dict
        Tree :func:`spectrum_to_tree` and :func:`add_moments` with node
        dictionaries.

    Returns
    -------
    numpy.ndarray
        A one-element object array containing the tree.
    """

    return np.array([tree])


def tree_to_numpy_predefined_size(
    tree, moment_names, max_n_nodes=15):
    """Convert a tree to a fixed-size NumPy array to be used by the apply_ufunc.

    Parameters
    ----------
    tree : dict
        Tree mapping node indices to dictionaries with bounds, threshold,
        parent, and moment values.
    moment_names : list of str
        Names of the moment variables to include.
    max_n_nodes : int, default=15
        Maximum number of nodes to retain in the output.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The node-value array and the corresponding variable names.
    """

    # unpack tree from array
    # only needed for the numpy of dtype dict version
    #tree = tree[0]
    
    variable_names = ['bounds_left', 'bounds_right', 'id_parent', 'thres']

    output = np.full((len(variable_names), max_n_nodes), -999.)
    indices = np.array(list(tree.keys()))
    indices = indices[indices < max_n_nodes]
    for i, var in enumerate(variable_names):
        if len(indices) > 0:
            output[i,indices] =  [tree[j][var] for j in indices]

    output_moments = np.full((len(moment_names), max_n_nodes), -999.)
    for i, var in enumerate(moment_names):
        if len(indices) > 0:
            pref, e = var.split('_')
            output_moments[i,indices] =  [tree[j]['moments'][pref][e] for j in indices]
    #print('output shapes', output.shape, output_moments.shape)

    return (np.concatenate((output, output_moments)),
            np.array(variable_names + moment_names))



def ufunc_wrapper(
        vel, inputvarnames, a, mask, 
        vel_step=None, params=None, var_peak=None, 
        meta=None
    ):
    """Wrap spectrum-to-tree processing for vectorized application.

    This helper is used by :func:`peakTree.ds_to_tree` through
    :func:`xarray.apply_ufunc` to convert one spectrum at a time into a
    fixed-size tree representation.

    Parameters
    ----------
    vel : numpy.ndarray
        Doppler velocity values associated with the input spectrum.
    inputvarnames : sequence of str
        Names of the variables stored along the input-variable dimension.
    a : numpy.ndarray
        Array containing the spectra and auxiliary variables. The selected
        variable is read from this array using ``var_peak``.
    mask : numpy.ndarray
        Boolean mask applied to the spectrum when the tree is built.
    vel_step : float, optional
        Velocity step size used by :func:`spectrum_to_tree`.
    params : dict, optional
        Parameters that control the tree-building routine, such as width and
        prominence thresholds.
    var_peak : str, optional
        Name of the variable in ``inputvarnames`` that should be used for peak finding.
    meta : dict, optional
        Mapping from variable names to the moment names that should be
        computed and stored in the tree.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        A fixed-size tree array and the corresponding variable names. The
        first element contains bounds, parent identifiers, thresholds, and
        moment values for the detected nodes; the second element lists the
        names associated with each row.

    See Also
    --------
    spectrum_to_tree : Build a tree from a single spectrum.
    add_moments : Attach moments to the tree nodes.
    ds_to_tree : Dataset-level entry point that applies this wrapper.
    """
    

    #print('wrapper start, type(a)', type(a))
    #print(inputvars, ' array shape', a.shape)

    tree = spectrum_to_tree(
        vel_step, a[np.argwhere(inputvarnames == var_peak)[0][0],:], 
        mask, params
    )

    tree = add_moments(
        tree, vel, a, inputvarnames, meta
    )
    #print(tree)

    moment_names = [f"{pref}_{e}" for pref, m in meta.items() for e in m]
    return tree_to_numpy_predefined_size(tree, moment_names)

    
