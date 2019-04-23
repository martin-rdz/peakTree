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
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

pTB = peakTree.peakTreeBuffer()
pTB = peakTree.peakTreeBuffer(system='Lacros')
#pTB.load_spec_file('data/D20170311_T2000_2100_Lim_zspc2nc_v1_02_standard.nc4')
# pTB.load_spec_file('data/D20170311_T2000_2100_Lim_zspc2nc_v1_02_standard_faster.nc4')
# pTB.load_spec_file('data/D20180122_T1030_1100_Lim_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')
#pTB.load_spec_file('data/D20180122_T1400_1600_Lim_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')
#pTB.load_spec_file('data/D20170130_T1937_2130_Lim_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')


# Punta Arenas
files = [
    'data/D20190207_T0700_0800_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190207_T0800_0900_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190207_T0900_1000_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190207_T1000_1100_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190210_T0400_0500_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190210_T0500_0600_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190210_T0600_0700_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190210_T0700_0800_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190211_T0900_1000_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190211_T1000_1100_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190211_T1100_1200_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190211_T1200_1300_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190212_T0400_0500_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190212_T0500_0600_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190212_T0600_0700_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190212_T0700_0800_Pun_zspc2nc_v1_02_standard.nc4',
]
files = [
    'data/D20190219_T0000_0100_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190219_T0100_0200_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190219_T0200_0300_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190219_T0300_0400_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190219_T0400_0500_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190219_T0500_0600_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190219_T0600_0700_Pun_zspc2nc_v1_02_standard.nc4',
    'data/D20190219_T0700_0800_Pun_zspc2nc_v1_02_standard.nc4',
]
files = [
    #'data/D20190222_T1100_1200_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T1200_1300_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T1300_1400_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T1400_1500_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T1500_1600_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T1600_1700_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T1700_1800_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T1800_1900_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T1900_2000_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T2000_2100_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T2100_2200_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T2200_2300_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190222_T2300_0000_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T0000_0100_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T0100_0200_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T0200_0300_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T0300_0400_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T0400_0500_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T0500_0600_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T0600_0700_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T0700_0800_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T0800_0900_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T0900_1000_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T1000_1100_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T1100_1200_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T1200_1300_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T1300_1400_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T1400_1500_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T1500_1600_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T1600_1700_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T1700_1800_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T1800_1900_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T1900_2000_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T2000_2100_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T2100_2200_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T2200_2300_Pun_zspc2nc_v1_02_standard.nc4',
    #'data/D20190223_T2300_0000_Pun_zspc2nc_v1_02_standard.nc4',

    'data/D20190317_T0600_0700_Pun_zspc2nc_v1_02_standard.nc4',
]

for f in files:
    pTB = peakTree.peakTreeBuffer(config_file='instrument_config.toml', system='Lacros_Pun')
    pTB.load_spec_file(f, load_to_ram=True)
    pTB.assemble_time_height('output/')
exit()

pTB = peakTree.peakTreeBuffer(system='Polarstern')
#pTB.load_spec_file('data/D20170605_T0030_0045_Pol_zspc2nc_v1_02_standard.nc4')
#pTB.load_spec_file('data/D20170629_T0830_0945_Pol_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')
#pTB.load_spec_file('data/D20170629_T0800_0930_Pol_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')
pTB.load_spec_file('data/D20170629_T0830_0945_Pol_zspc2nc_v1_02_standard.nc4')
pTB.assemble_time_height('output/')


pTB = peakTree.peakTreeBuffer(system='Lindenberg')
#pTB.load_spec_file('data/D20150801_T1500_1800_Lin_zspc2nc_v1_02_standard.nc4')
#pTB.assemble_time_height('output/')

pTB = peakTree.peakTreeBuffer(system='Lacros_at_ACCEPT')
#pTB.load_spec_file('data/D20141005_T1600_1700_Cab_zspc2nc_v1_02_standard.nc4')
#pTB.load_spec_file('data/D20141006_T1500_1600_Cab_zspc2nc_v1_02_standard.nc4')
#pTB.load_spec_file('data/D20141007_T1700_1800_Cab_zspc2nc_v1_02_standard.nc4')
pTB.load_spec_file('data/D20141012_T1600_1700_Cab_zspc2nc_v1_02_standard.nc4')
#pTB.load_spec_file('data/D20141115_T1600_1700_Cab_zspc2nc_v1_02_standard.nc4')
#pTB.load_spec_file('data/D20141115_T1700_1800_Cab_zspc2nc_v1_02_standard.nc4')

#pTB.load_spec_file('data/D20141020_T1600_1700_Cab_zspc2nc_v1_02_standard.nc4')
#pTB.load_spec_file('data/D20141020_T1700_1800_Cab_zspc2nc_v1_02_standard.nc4')
#pTB.load_spec_file('data/D20141020_T1800_1900_Cab_zspc2nc_v1_02_standard.nc4')

# Heikes cases
files = [
    'data/D20141005_T1600_1700_Cab_zspc2nc_v1_02_standard.nc4', #added by mr
    'data/D20141012_T1400_1500_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141012_T1500_1600_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141012_T1600_1700_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141012_T1700_1800_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141012_T1800_1900_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141016_T0200_0300_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141019_T1400_1500_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141019_T1500_1600_Cab_zspc2nc_v1_02_standard.nc4',
    #'data/D20141102_T1400_1500_Cab_zspc2nc_v1_02_standard.nc4',
    #'data/D20141102_T1500_1600_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141102_T1600_1700_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141102_T1700_1800_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141102_T1800_1900_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141102_T1900_2000_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141117_T0000_0100_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141117_T0100_0200_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141117_T0200_0300_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141117_T2000_2100_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141117_T2100_2200_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141117_T2200_2300_Cab_zspc2nc_v1_02_standard.nc4',
    'data/D20141117_T2300_0000_Cab_zspc2nc_v1_02_standard.nc4',
]

files = [
    'data/D20141102_T1600_1700_Cab_zspc2nc_v1_02_standard.nc4',
#    'data/D20141102_T1700_1800_Cab_zspc2nc_v1_02_standard.nc4',
#    'data/D20141102_T1800_1900_Cab_zspc2nc_v1_02_standard.nc4',
#    'data/D20141102_T1900_2000_Cab_zspc2nc_v1_02_standard.nc4',
]
exit()

for f in files:
    pTB = peakTree.peakTreeBuffer(system='Lacros_at_ACCEPT')
    pTB.load_spec_file(f)
    pTB.assemble_time_height('output/')
