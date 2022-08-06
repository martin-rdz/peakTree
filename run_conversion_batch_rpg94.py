#! /usr/bin/env python3
# coding=utf-8

import datetime
import os
import re
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

path = f"data/{args.instrument}/Y{date.strftime('%Y')}/M{date.strftime('%m')}/D{date.strftime('%d')}/"

files = os.listdir(path)
files = [f for f in files if ('.LV0' in f[-4:])]
print(files)
#files = [f for f in files if int(re.findall("T(\d*)", f)[0]) > 1300]
#files = [f for f in files if int(re.findall("_(\d*)_", f)[0]) < 60000]

outpath = f"output/{args.instrument}/{date.strftime('%Y%m%d')}/"
if not os.path.isdir(outpath):
    print('create output path ', outpath)
    os.mkdir(outpath)

pTB = peakTree.peakTreeBuffer(config_file='instrument_config.toml', system=args.config)

print('doing only ', files)

for f in sorted(files)[:]:
    print('now doing ', f, outpath)
    pTB.load(path+f, load_to_ram=True)
    pTB.assemble_time_height(outpath)


