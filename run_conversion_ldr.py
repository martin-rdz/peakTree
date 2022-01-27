#! /usr/bin/env python3
# coding=utf-8

import datetime
import os
#import matplotlib
#matplotlib.use('Agg')
#import numpy as np
#import matplotlib.pyplot as plt
import argparse

#import sys, os
import peakTree
import peakTree.helpers as h

import logging
log = logging.getLogger('peakTree')
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

parser = argparse.ArgumentParser(description='peakTree conversion')
parser.add_argument('--date', help='date in the format YYYYMMDD', required=True)
parser.add_argument('--instrument', help='radar model NMRA, MBR5', default='NMRA')
parser.add_argument('--config', help='config entry to use', default='Lacros_Pun')

args = parser.parse_args()
date = datetime.datetime.strptime(args.date, '%Y%m%d')

#date = datetime.datetime(2019, 2, 3)

path = 'data/{}/{}/'.format(args.instrument, date.strftime('%Y%m%d'))

files = os.listdir(path)
files = [f for f in files if '.nc' in f]
print(files)

outpath = 'output/{}/{}/'.format(args.instrument, date.strftime('%Y%m%d'))
if not os.path.isdir(outpath):
    print('create output path ', outpath)
    os.mkdir(outpath)

pTB = peakTree.peakTreeBuffer(config_file='instrument_config.toml', system=args.config)

# filter for patrics test
files = [f for f in files if "T1300_" in f]
files = [f for f in files if "v2.0" in f]
print(files)

for f in files[:]:
    print('now doing ', f)
    if args.config == 'kazr_mosaic':
        pTB.load_newkazr_file(path+f, load_to_ram=True)
    else:
        pTB.load_spec_file(path+f, load_to_ram=True)
    pTB.assemble_time_height(outpath)


exit()



pTB = peakTree.peakTreeBuffer(config_file='instrument_config.toml', system='Lacros_Pun')
pTB.load_spec_file('data/D20190317_T0600_0700_Pun_zspc2nc_v1_02_standard.nc4', load_to_ram=True)
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
