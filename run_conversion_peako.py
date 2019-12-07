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

pTB = peakTree.peakTreeBuffer()
pTB = peakTree.peakTreeBuffer(system='kazr_baecc_peako')
#pTB.load_peako_file('data/kazr20140221220005_22.54_22.77_spectra_edges.nc')
pTB.load_peako_file('data/kazr20140202160004_16.001_16.99_spectra_edges.nc')
#dt = datetime.datetime(2014,2,21,22,40,0,0)
#pTB.get_tree_at(h.dt_to_ts(dt), 2900)
#pTB.assemble_time_height('output/')


pTB = peakTree.peakTreeBuffer(system='kazr_baecc')
#pTB.load_kazr_file('data/20140221_2200_kazr_preprocessed.nc4')
pTB.load_kazr_file('data/20140202_1600_kazr_preprocessed.nc4')
#dt = datetime.datetime(2014,2,21,22,40,0,0)
#pTB.get_tree_at(h.dt_to_ts(dt), 2900, temporal_average=4)
pTB.assemble_time_height('output/')
