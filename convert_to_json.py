#! /usr/bin/env python3
# coding=utf-8

import datetime
import numpy as np

import argparse
import sys, os
import json
import peakTree
import peakTree.helpers as h
import peakTree.VIS_Colormaps as VIS_Colormaps

parser = argparse.ArgumentParser(description='Convert to the json')
parser.add_argument('file', help='path of the netcdf file')
parser.add_argument('output', help='name of the output json/js (inside the output folder)')
parser.add_argument('--time-interval', type=str, default='0-max', help='time to convert. e.g. 0-400, 0-max, 500-max')
parser.add_argument('--range-interval', type=str, default='0-350', help='range to convert. e.g. 0-7000, 0-max, 500-max')
args = parser.parse_args()

jsfunct = 'function get_data () {return data}\nvar data =' 

def format_for_json(elem):
    if isinstance(elem, np.integer):
        return int(elem)
    elif isinstance(elem, np.floating):
        return round(float(elem), 4)
    elif isinstance(elem, np.ndarray):
        return elem.tolist()
    elif isinstance(elem, float):
        return round(elem, 4)
    else:
        return elem

def get_tree_region(pTB, it_range, ir_range):
    temp_avg = 10
    trees = {}
    var = np.empty((it_range[1]-it_range[0], ir_range[1]-ir_range[0]), dtype=object)
    for it in range(*it_range):
        print('time step ', it, ' from ', it_range)
        for ir in range(*ir_range):
            #print('it',it, it-it_range[0], 'ir', ir, ".ir", ir_range[1]-ir-1, pTB.range[ir])
            travtree, _ = pTB.get_tree_at((it, pTB.timestamps[it]), (ir, pTB.range[ir]), 
                                          silent=True)
            nodes = {}
            for k, v in travtree.items():
                v['id'] = k
                v['bounds'] = list(map(int, v['bounds']))
                if v['parent_id'] == -1:
                    del v['parent_id']
                v['z'] = h.lin2z(v['z'])
                v['width'] = v['width'] if np.isfinite(v['width']) else -99
                v['skew'] = v['skew'] if np.isfinite(v['skew']) else -99
                if 'ldr' in v:
                    v['ldr'] = h.lin2z(v['ldr']) if np.isfinite(h.lin2z(v['ldr'])) else -99
                    v['ldrmax'] = h.lin2z(v['ldrmax']) if np.isfinite(h.lin2z(v['ldrmax'])) else -99
                else:
                    v['ldr'] = -99
                    v['ldrmax'] = -99
                v['thres'] = h.lin2z(v['thres'])
                v = {ky: format_for_json(val) for ky, val in v.items()}
                nodes[k] = v
                
            #k = '{:0>2d}.{:0>2d}'.format(it-it_range[0], ir_range[1]-ir-1)
            #trees[k] = nodes
            var[it,ir] = nodes

    timestamps = pTB.timestamps[it_range[0]:it_range[1]]
    print(pTB.range[ir_range[0]], pTB.range[ir_range[1]])
    ranges = pTB.range[ir_range[0]:ir_range[1]]
    print("ranges ", ranges)

    meta = {}
    for a in pTB.f.ncattrs():
        meta[a] = pTB.f.getncattr(a)
    meta['json_timeinterval'] = it_range
    meta['json_rangeinterval'] = ir_range

    return {'ts': format_for_json(timestamps), 'rg': format_for_json(ranges), 
            'var': var.tolist(), 'dimlabel': ['time', 'range', 'tree'],
            'paraminfo': meta}


pTB = peakTree.peakTreeBuffer()
#pTB = peakTree.peakTreeBuffer(system="Polarstern")
#pTB.load_peakTree_file('output/20170629_0830_Pol_peakTree.nc4')
pTB.load_peakTree_file(args.file)


time_interval = args.time_interval.split('-')
if time_interval[1] == "max":
    time_interval[1] = pTB.timestamps.shape[0][-1]
time_interval = list(map(int, time_interval))

range_interval = args.range_interval.split('-')
if range_interval[1] == "max":
    range_interval[1] = pTB.range[0][-1]
range_interval = list(map(int, range_interval))

json_data = get_tree_region(pTB, time_interval, range_interval)

with open(args.output + '_container.json', 'w') as outfile:
    json_string = json.dumps(json_data)
    outfile.write(json_string)

with open(args.output + '_container.js', 'w') as outfile:
    json_string = json.dumps(json_data)
    json_string = jsfunct + json_string + ";"
    outfile.write(json_string)