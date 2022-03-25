#! /usr/bin/env python3
# coding=utf-8

import datetime
#import matplotlib
#matplotlib.use('Agg')
#import numpy as np
#import matplotlib.pyplot as plt

#import sys, os
import peakTree
import peakTree.helpers as h

import logging
log = logging.getLogger('peakTree')
#log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

#pTB = peakTree.peakTreeBuffer()
#pTB = peakTree.peakTreeBuffer(system='Lacros')
#pTB.load_spec_file('data/D20170311_T2000_2100_Lim_zspc2nc_v1_02_standard.nc4')
# pTB.load_spec_file('data/D20170311_T2000_2100_Lim_zspc2nc_v1_02_standard_faster.nc4')
# pTB.load_spec_file('data/D20180122_T1030_1100_Lim_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')
#pTB.load_spec_file('data/D20180122_T1400_1600_Lim_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')
#pTB.load_spec_file('data/D20170130_T1937_2130_Lim_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')


# for f in files:
#     pTB = peakTree.peakTreeBuffer(config_file='instrument_config.toml', system='Lacros_Pun')
#     pTB.load_spec_file(f, load_to_ram=True)
#     pTB.assemble_time_height('output/')


#pTB = peakTree.peakTreeBuffer(system='Polarstern')
#pTB.load_spec_file('data/D20170605_T0030_0045_Pol_zspc2nc_v1_02_standard.nc4')
#pTB.load_spec_file('data/D20170629_T0830_0945_Pol_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')
#pTB.load_spec_file('data/D20170629_T0800_0930_Pol_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')
#pTB.load_spec_file('data/D20170629_T0830_0945_Pol_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')

#pTB = peakTree.peakTreeBuffer(config_file='instrument_config.toml', system='Lacros_Pun')
pTB = peakTree.peakTreeBuffer(system='Lacros_Pun')
#pTB.load_spec_file('data/D20190317_T0600_0700_Pun_zspc2nc_v1_02_standard.nc4', load_to_ram=True)
#pTB.load_spec_file('data/D20190911_T0300_0400_Pun_zspc2nc_v2.0_standard.nc4', load_to_ram=True)
pTB.load_spec_file('data/D20190313_T0800_0900_Pun_zspc2nc_v2.0_standard.nc4', load_to_ram=True)
pTB.assemble_time_height('output/')
exit()


path = 'data/'
files = os.listdir(path)
valid_time = ['20190711', '20190713']
print('total no files in ', path, ' : ', len(files))
files = [path+f for f in files if f[1:9] >= valid_time[0] and f[1:9] <= valid_time[1]]
files = sorted(files)
print('files selected', len(files))

for f in files:
    pTB = peakTree.peakTreeBuffer(system='Lacros_at_ACCEPT')
    pTB.load_spec_file(f)
    pTB.assemble_time_height('output/')
