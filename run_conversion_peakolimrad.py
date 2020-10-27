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
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

pTB = peakTree.peakTreeBuffer(system='limrad_peako')

# IDEA: for now run with temporal_average = False
# t_avg: number of neighbors in time dimension (both sides)
# h_avg: number of neighbors in range dimension (both sides)
# span: loess span
# width_thres: minimum peak width [m/s]
# prom_thres: minimum peak prominence in dBZ

#pTB.load_limrad_spec('data/20181216-1210-1215_LIMRAD94_spectra.nc', load_to_ram=True)
#pTB.load_limrad_spec('data/20181216-1510-1515_LIMRAD94_spectra.nc', load_to_ram=True)
pTB.load_limrad_spec('data/20190223-1440-1500_LIMRAD94_spectra.nc', load_to_ram=True)

pTB.assemble_time_height('output/')
