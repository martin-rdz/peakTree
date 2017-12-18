#! /usr/bin/env python3
# coding=utf-8

import datetime
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

import sys, os
import peakTree
import peakTree.helpers as h

ts = h.dt_to_ts(datetime.datetime(2017,3,11,20,41))
rg = 3500

pTB = peakTree.peakTreeBuffer()
pTB.load_spec_file('data/D20170311_T2000_2100_Lim_zspc2nc_v1_02_standard.nc4')
tree, spectrum = pTB.get_tree_at(ts, rg)

print('tree is a dictionary of dictionaries')
print(tree)
print('pretty representation of the tree')
print(peakTree.print_tree.travtree2text(tree))

peakTree.print_tree.plot_spectrum(tree, spectrum, 'plots/test_profile/')

# for rg in pTB.range[:300]:
# for rg in pTB.range[:300]:
#    tree, spectrum = pTB.get_tree_at(ts, rg)
#    peakTree.print_tree.plot_spectrum(tree, spectrum, 'plots/test_profile/')