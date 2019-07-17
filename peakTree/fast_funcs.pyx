
cimport cython
cimport numpy as np
import numpy as np

#@# distutils: define_macros=CYTHON_TRACE_NOGIL=1


# @cython.boundscheck(False)  # Deactivate bounds checking
# #@cython.wraparound(False)   # Deactivate negative indexing.
# #@profile
# def get_minima(double[:] array):
#     """get the minima of an array by calculating the derivative

#     tested against scipy.signal.argrelmin without difference
#     in result or speed

#     Returns:
#         list of ``(index, value at index)``
#     """
#     #sdiff = np.ma.diff(np.sign(np.ma.diff(array)))
#     cdef long[:] sdiff = np.diff(np.sign(np.diff(array)))
#     rising_1 = np.where(sdiff == 2)
#     rising_2 = (sdiff[:-1] == 1) & (sdiff[1:] == 1)
#     rising_all = rising_1
#     rising_all[1:] = rising_all[1:] | rising_2
#     min_ind = np.where(rising_all)[0] + 1
#     minima = list(zip(min_ind, array[min_ind]))
#     return sorted(minima, key=lambda x: x[1])


cdef fill_with(array, np.uint8_t[:] mask, double fill):
    """fill array with fill value where mask is True"""
    cdef double[:] filled = array.copy()
    cdef Py_ssize_t length = array.shape[0]
    cdef int i

    for i in range(length):
        if mask[i]:
            filled[i] = fill
    return filled


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def detect_peak_simple(double[:] array, double lthres):
    """detect noise separated peaks

    Args:
        array: with Doppler spectrum
        lthres: threshold
    Returns:
        list of indices (as tuple)
    """
    cdef list peakindices = []
    
    cdef int first_index = -1
    cdef int current_index = -1
    cdef int i
    cdef Py_ssize_t length = array.shape[0]

    for i in range(length):
        if array[i] > lthres:
            if first_index == -1:
                first_index = i
                current_index = i
            else:
                current_index = i
        else:
            # write out previous run
            if  first_index != -1 and current_index != -1:
                peakindices.append((first_index, current_index))
            first_index = -1
            current_index = -1

    # fix if run goes to last element
    if first_index != -1 and current_index != -1:
         peakindices.append((first_index, current_index))
            
    return peakindices


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_minima(double[:] array):
    """rewritten to a for loop in cython

    Returns:
        list of ``(index, value at index)``
    """

    cdef Py_ssize_t length = array.shape[0]
    cdef int i
    cdef list minima = []

    for i in range(1,length-1):
        # print(i, array[i])
        if (array[i] < array[i-1] and array[i] < array[i+1]):
            minima.append((i, array[i]))
        # account for 2-bin wide minimum [... 3 2 2 4 ...]
        # and [... 3 2 2 1 4 ...]
        if array[i] == array[i-1]:
            if i > 2 and i < length-2 \
              and array[i] < array[i+1] and array[i] < array[i-2]: 
                minima.append((i, array[i]))


    # sdiff = np.diff(np.sign(np.diff(array)))
    # rising_1 = (sdiff == 2)
    # rising_2 = (sdiff[:-1] == 1) & (sdiff[1:] == 1)
    # rising_all = rising_1
    # rising_all[1:] = rising_all[1:] | rising_2
    # min_ind = np.where(rising_all)[0] + 1
    #minima = list(zip(min_ind, array[min_ind]))
    #return sorted(minima, key=lambda x: x[1])
    return sorted(minima, key=lambda x: x[1])



cdef int full_tree_id(list coord):
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

    cdef int idx = 0
    cdef int N = len(coord)
    cdef int i

    for i in range(1,N):
    #for ind, flag in enumerate(reversed(coord)):
        if coord[i] == 1:
            idx = (2*idx + 2)
        else:
            idx = (2*idx + 1)
    #print('new', coord,'->',idx)
    return idx


cpdef dict coords_to_id(list traversed):
    """calculate the id in level-order from the coordinates

    Args:
        input: traversed tree as list of dict
    Returns:
        traversed tree (dict) with id as key
    """
    cdef dict traversed_id = {}  
    cdef int L = len(traversed)
    cdef int i
    cdef int k
    cdef int parent

    for i in range(L):
        k = full_tree_id(traversed[i]['coords'])
        traversed_id[k] = traversed[i]
        parent = np.floor((k-1)/2.)
        traversed_id[k]['parent_id'] = parent if k != 0 else -1

    return traversed_id



cdef class Node(object):
    """class to generate the tree
    
    Args:
        bounds: boundaries in bin coordinates
        spec_chunk: spectral reflectivity within this node
        noise_thres: noise threshold hat separated this peak
        root: flag indicating if root node
        parent_lvl: level of the parent node
    """
    cdef public tuple bounds
    cdef public list children
    cdef public int level
    cdef public bint root
    cdef public float threshold
    cdef public np.ndarray spec
    cdef public float prom_filter


    def __init__(self, bounds, spec_chunk, noise_thres, root=False, parent_lvl=0):
        self.bounds = bounds
        self.children = []
        self.level = 0 if root else parent_lvl + 1
        self.root = root
        self.threshold = noise_thres
        self.spec = spec_chunk
        # faster to have prominence filter in linear units
        self.prom_filter = 10**(1./10)
        # prominence filter  2dB (Shupe 2004) or even 6 (Williams 2018)
        #print('at node ', bounds, h.lin2z(noise_thres), spec_chunk)

    cpdef int add_noise_sep(self, bounds_left, bounds_right, thres):
        """add a nose separated peak/node
        
        Args:
            bounds_left: boundaries of the left peak
            bounds_right: boundaries of the right peak
            thres: threshold that separates the peaks
        """
        if len(self.children) > 0 and self.children[0].bounds[0] <= bounds_left[0] and self.children[0].bounds[1] >= bounds_right[1]:
            # append to left child
            self.children[0].add_noise_sep(bounds_left, bounds_right, thres)
        elif len(self.children) > 0 and self.children[1].bounds[0] <= bounds_left[0] and self.children[1].bounds[1] >= bounds_right[1]:
            # append to right child
            self.children[1].add_noise_sep(bounds_left, bounds_right, thres)
        else:
            # insert here
            spec_left = self.spec[bounds_left[0]-self.bounds[0]:bounds_left[1]+1-self.bounds[0]]
            spec_right = self.spec[bounds_right[0]-self.bounds[0]:bounds_right[1]+1-self.bounds[0]]
            prom_left = spec_left[spec_left.argmax()]/thres
            prom_right = spec_right[spec_right.argmax()]/thres
            if prom_left > self.prom_filter and prom_right > self.prom_filter:
                self.children = [Node(bounds_left, spec_left, thres, parent_lvl=self.level), 
                                 Node(bounds_right, spec_right, thres, parent_lvl=self.level)]
            else:
                #print('omitted noise sep. peak at ', bounds_left, bounds_right, 10*np.log10(prom_left), 10*np.log10(prom_right))
                pass
        return 0

    cpdef int add_min(self, new_index, current_thres, ignore_prom=False):
        """add a local minimum

        Args:
            new_index: bin index of minimum
            current_threshold: reflectivity that separates the peaks
            ignore_prom (optional): ignore the prominence threshold
        """
        if new_index < self.bounds[0] or new_index > self.bounds[1]:
            raise ValueError("child out of parents bounds")

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

            if (prom_left > self.prom_filter and prom_right > self.prom_filter) or ignore_prom:
                self.children.append(Node((self.bounds[0], new_index), 
                                     spec_left, current_thres, parent_lvl=self.level))
                self.children.append(Node((new_index, self.bounds[1]), 
                                     spec_right, current_thres, parent_lvl=self.level))
        return 0


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef moment(double[:] x, double[:] Z):
    """mean, rms, skew for a vel, Z part of the spectrum
    
    Args:
        x: velocity of bin
        Z: spectral reflectivity
    Returns:
        dict with v, width, skew
    """
    cdef double sumZ = np.sum(Z) # memory over processing time
    # print('x fast', np.array(x)[:5], np.array(x)[-5:])
    # print('z fast', np.array(Z)[:5], np.array(Z)[-5:])
    cdef double mean = np.sum(np.multiply(x, Z))/sumZ
    # print('mean fast', mean)
    cdef double[:] x_mean = np.zeros(x.shape[0])
    cdef Py_ssize_t length = x.shape[0]
    cdef int i
    for i in range(length):
        x_mean[i] = x[i]-mean # memory over processing time
    cdef double rms = np.sqrt(np.sum(np.multiply(np.power(x_mean, 2),Z))/sumZ)
    cdef double skew = np.sum(np.multiply(np.power(x_mean,3),Z)/(sumZ*(rms*rms*rms)))
    return mean, rms, skew


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
# moved to def for line profiling issues
cpdef calc_moments(dict spectrum, tuple bounds, double thres, bint no_cut=False):
    """calc the moments following the formulas given by Görsdorf2015 and Maahn2017

    Args:
        spectrum: spectrum dict
        bounds: boundaries (bin no)
        thres: threshold used
    Returns
        moment, spectrum
    """
    cdef double[:] vel = spectrum['vel']
    cdef double[:] specZ = spectrum['specZ']
    cdef np.uint8_t[:] specZ_mask = spectrum['specZ_mask'].astype(np.uint8)

    cdef double[:] masked_Z = np.zeros(bounds[1] - bounds[0])


    cdef double Z = np.sum(specZ[bounds[0]:bounds[1]+1])
    # print('tres', thres)
    # print('thres fast', np.array(spectrum['specZ'][bounds[0]:bounds[1]+1]<thres)[:5], np.array(spectrum['specZ'][bounds[0]:bounds[1]+1]<thres)[-5:])
    # print('thres fast', np.array(spectrum['specZ_mask'][bounds[0]:bounds[1]+1])[:5], np.array(spectrum['specZ_mask'][bounds[0]:bounds[1]+1])[-5:])
    cdef np.uint8_t[:] Z_mask = np.logical_or(np.less(specZ[bounds[0]:bounds[1]+1], thres), specZ_mask[bounds[0]:bounds[1]+1]).astype(np.uint8)

    if not no_cut:
        masked_Z = fill_with(specZ[bounds[0]:bounds[1]+1], Z_mask, 0.0)
    else:
        masked_Z = fill_with(specZ, specZ_mask, 0.0)
        masked_Z[:bounds[0]] = 0.0
        masked_Z[bounds[1]+1:] = 0.0

    # seemed to have been quite slow (84us per call or 24% of function)
    cdef tuple mom = moment(vel[bounds[0]:bounds[1]+1], masked_Z)
    cdef dict moments = {'v': mom[0], 'width': mom[1], 'skew': mom[2]}
    moments['z'] = Z
    
    #spectrum['specZco'] = spectrum['specZ']/(1+spectrum['specLDR'])
    #spectrum['specZcx'] = (spectrum['specLDR']*spectrum['specZ'])/(1+spectrum['specLDR'])

    cdef double[:] specSNRco = spectrum['specSNRco']
    cdef int ind_max = np.argmax(specSNRco[bounds[0]:bounds[1]+1])


    cdef double prominence = specZ[bounds[0]:bounds[1]+1][ind_max] / thres
    cdef np.uint8_t prominence_mask = specZ_mask[bounds[0]:bounds[1]+1][ind_max]
    if not prominence_mask:
        moments['prominence'] = prominence
    else:
        moments['prominence'] = 1e-99
    #assert np.all(validSNRco.mask == validSNRcx.mask)
    #print('SNRco', h.lin2z(spectrum['validSNRco'][bounds[0]:bounds[1]+1]))
    #print('SNRcx', h.lin2z(spectrum['validSNRcx'][bounds[0]:bounds[1]+1]))
    #print('LDR', h.lin2z(spectrum['validSNRcx'][bounds[0]:bounds[1]+1]/spectrum['validSNRco'][bounds[0]:bounds[1]+1]))

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

    cdef double[:] specLDR = spectrum['specLDR']
    cdef double[:] specZcx_validcx = spectrum['specZcx_validcx']
    cdef double[:] specZ_validcx = spectrum['specZ_validcx']
    cdef np.uint8_t[:] specZcx_mask = spectrum['specZcx_mask'].astype(np.uint8)

    cdef double ldrmax = specLDR[bounds[0]:bounds[1]+1][ind_max]
    cdef double ldr2
    if not np.all(specZcx_mask):
        ldr2 = np.sum(specZcx_validcx[bounds[0]:bounds[1]+1]) / np.sum(specZ_validcx[bounds[0]:bounds[1]+1])
    else:
        ldr2 = np.nan

    #cdef double[:] decoup = 10**(spectrum['decoupling']/10.)
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


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def unsave_mean_axis_1(float[:,:] arr):
    cdef Py_ssize_t N = arr.shape[0]
    cdef Py_ssize_t M = arr.shape[1]
    result = np.zeros(N)
    cdef double[:] result_v = result
    cdef int m,n 
    cdef double mean

    for n in range(N):
        mean = 0.0
        for m in range(M):
            mean += arr[n,m]
        
        result_v[n] = mean/M
    
    return result_v

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def unsave_mean1d(float[:] arr):
    cdef Py_ssize_t  N = arr.shape[0]
    cdef float result = 0
    cdef int n
    for n in range(N):
        result += arr[n]
        
    result = result/N
    return result



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef load_spec_mira(float[:,:,:] Z, float[:,:,:] LDR, float[:,:,:] SNRco, int ir, int it_b, int it_e):

    cdef Py_ssize_t no_bins = Z.shape[0]
    cdef int no_averages = it_e - it_b
    specZ = np.zeros(no_bins)
    cdef double[:] specZ_v = specZ
    specZcx = np.zeros(no_bins)
    cdef double[:] specZcx_v = specZcx
    specLDR = np.zeros(no_bins)
    cdef double[:] specLDR_v = specLDR
    specSNRco = np.zeros(no_bins)
    cdef double[:] specSNRco_v = specSNRco
    cdef int b

    for b in range(no_bins):
        specZ_v[b] = unsave_mean1d(Z[b,ir,it_b:it_e+1])
        #specZcx[b] = unsave_mean1d(LDR[b,ir,it_b:it_e+1])
        specZcx_v[b] = unsave_mean1d(np.multiply(Z[b,ir,it_b:it_e+1], LDR[b,ir,it_b:it_e+1]))
        if not specZ[b] == 0:
            specLDR_v[b] = specZcx_v[b]/specZ_v[b]
        else:
            specLDR_v[b] = 0
        specSNRco_v[b] = unsave_mean1d(SNRco[b,ir,it_b:it_e+1])

    return np.asarray(specZ_v), np.asarray(specZcx_v), np.asarray(specLDR_v), np.asarray(specSNRco_v), no_averages