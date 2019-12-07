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
 
pTB = peakTree.peakTreeBuffer(system='joyrad_nya') 
#pTB.load_joyrad_file('data/joyrad94_nya_20191108000001_P01_ZEN.nc', load_to_ram=True) 
#TB.load_joyrad_file('data/joyrad94_nya_20191108120000_P01_ZEN.nc', load_to_ram=True)

# older files
pTB.load_joyrad_file('data/joyrad94_nya_20170602100002_P05_ZEN.nc', load_to_ram=True)
#pTB.load_joyrad_file('data/joyrad94_nya_20170602110001_P05_ZEN.nc', load_to_ram=True)
pTB.assemble_time_height('output/') 
exit()


pTB.load_joyrad_file('data/joyrad94_nya_20191108000001_P01_ZEN.nc', load_to_ram=True) 
pTB.assemble_time_height('output/') 