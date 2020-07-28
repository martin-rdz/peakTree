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
 
pTB = peakTree.peakTreeBuffer(system='kazr_mosaic') 
pTB.load_newkazr_file('data/kazr_mosaic_ge/moskazrcfrspcgecopolM1.a1.20191021.020003.nc', load_to_ram=True) 
pTB.assemble_time_height('output/kazr_mosaic_ge/') 





#pTB = peakTree.peakTreeBuffer(system='kazr_baecc') 
#pTB.load_kazr_file('data/tmpkazrspeccmaskgecopolM1.a0.20140202.160004.cdf', load_to_ram=True) 
#pTB.load_kazr_file('data/tmpkazrspeccmaskgecopolM1.a0.20140221.220005.cdf', load_to_ram=True)
#dt = datetime.datetime(2014,2,21,22,40,0,0) 
#pTB.get_tree_at(h.dt_to_ts(dt), 2900, temporal_average=4) 
#pTB.assemble_time_height('output/') 
