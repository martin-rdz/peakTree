#! /usr/bin/env python3
# coding=utf-8
"""
Author: radenz@tropos.de

"""

import sys, os
import pickle
sys.path.append("../peakTree") 
import peakTree as pT
import peakTree.helpers as h
import peakTree.print_tree as print_tree
import numpy as np

def test_dummy():
    ts = 1489262404
    rg = 3300
    pTB = pT.peakTreeBuffer()
    pTB.load_spec_file('data/D20170311_T2000_2100_Lim_zspc2nc_v1_02_standard.nc4')
    tree_spec, _ = pTB.get_tree_at(ts, rg)



def test_simple_tree():
    #ts = h.dt_to_ts(datetime.datetime(2017,3,11,20,24))
    #rg = 3500
    tree = {0: {'coords': [0], 'thres': 6.802836165522168e-06, 'width': 0.26523734151119865, 'prominence': 3879.4732574640866, 
                'ldr': 0.0055540428, 'bounds': (218, 249), 'skew': -1.906577433283317, 'ldrmax': 0.001390509, 'parent_id': -1, 
                'v': -1.0857420450595399, 'z': 0.16743573904022924}, 
            1: {'coords': [0, 0], 'thres': 7.6713631642633118e-05, 'width': 0.10727406506091237, 'prominence': 29.95259218616123, 
                'ldr': 0.013112152, 'bounds': (218, 229), 'skew': 0.40514998741269959, 'ldrmax': 0.0018346846, 'parent_id': 0, 
                'v': -1.9609763737501178, 'z': 0.00929530537177925}, 
            2: {'coords': [0, 1], 'thres': 7.6713631642633118e-05, 'width': 0.16158672836317653, 'prominence': 344.02518058323699, 
                'ldr': 0.0017749886, 'bounds': (229, 249), 'skew': -0.1581213963485647, 'ldrmax': 0.001390509, 'parent_id': 0, 
                'v': -1.0346927955089034, 'z': 0.15821714730009262}}
    
    # test the traversal in print_tree
    for var in ['bounds', 'coords']:
        print(var)
        print([node[var] for node in print_tree.iternodes(tree)])
        print([node[var] for node in tree.values()])
        assert [node[var] for node in print_tree.iternodes(tree)] == [node[var] for node in tree.values()]


    output_lines = [([-1.9609763737501178, -1.0857420450595399], [0.00929530537177925, 0.16743573904022924]), ([-1.0346927955089034, -1.0857420450595399], [0.15821714730009262, 0.16743573904022924])]
    for pair in zip(print_tree.gen_lines_to_par(tree), output_lines):
        print(pair)
        assert pair[0] == pair[1]

    # check travtree2text, no real idea yet
    #print(print_tree.travtree2text(tree).split('\n')[1:])


def test_complex_tree():
    #ts = h.dt_to_ts(datetime.datetime(2017,3,11,20,41))
    #rg = 3300
    tree = {0: {'coords': [0], 'thres': 5.394444479476282e-06, 'width': 0.33711132185863252, 'prominence': 1202.5028033247074, 
                'ldr': 0.0038949186, 'bounds': (223, 249), 'skew': -0.49973588804924224, 'ldrmax': 0.0020721117, 'parent_id': -1, 
                'v': -1.1118590290685955, 'z': 0.057759765141895514}, 
            1: {'coords': [0, 0], 'thres': 0.00048636521387379616, 'width': 0.086520710596273476, 'prominence': 7.5628031111610969, 
                'ldr': 0.0022399588, 'bounds': (223, 232), 'skew': 0.21021876089117944, 'ldrmax': 0.0014006387, 'parent_id': 0, 
                'v': -1.6225984604684851, 'z': 0.013663957421385931}, 
            2: {'coords': [0, 1], 'thres': 0.00048636521387379616, 'width': 0.19793260676718102, 'prominence': 13.337373693491468, 
                'ldr': 0.0040453696, 'bounds': (232, 249), 'skew': -0.34050094500845968, 'ldrmax': 0.0020721117, 'parent_id': 0, 
                'v': -0.96328200098985761, 'z': 0.044582172934383379}, 
            5: {'coords': [0, 1, 0], 'thres': 0.0024113187100738287, 'width': 0.065579493685471807, 'prominence': 1.6204062700507458, 
                'ldr': 0.0022046745, 'bounds': (232, 238), 'skew': 0.17177819230016397, 'ldrmax': 0.0016024747, 'parent_id': 2, 
                'v': -1.1526232128808762, 'z': 0.016210198467888404}, 
            6: {'coords': [0, 1, 1], 'thres': 0.0024113187100738287, 'width': 0.092693976042551807, 'prominence': 2.690160608736416, 
                'ldr': 0.0045316913, 'bounds': (238, 249), 'skew': -0.32987259931376162, 'ldrmax': 0.0020721117, 'parent_id': 2, 
                'v': -0.8741997376771532, 'z': 0.030783293176568804}}

    # test the traversal in print_tree
    for var in ['bounds', 'coords']:
        print(var)
        print([node[var] for node in print_tree.iternodes(tree)])
        print([node[var] for node in tree.values()])
        assert [node[var] for node in print_tree.iternodes(tree)] == [node[var] for node in tree.values()]

    print(print_tree.travtree2text(tree))


def test_coord_to_binary():
    assert pT.full_tree_id([0]) == 0
    assert pT.full_tree_id([0, 0]) == 1
    assert pT.full_tree_id([0, 1]) == 2
    assert pT.full_tree_id([0, 0, 0]) == 3
    assert pT.full_tree_id([0, 0, 1]) == 4
    assert pT.full_tree_id([0, 1, 0]) == 5
    assert pT.full_tree_id([0, 1, 1]) == 6
    assert pT.full_tree_id([0, 0, 0, 0]) == 7
    assert pT.full_tree_id([0, 0, 1, 1]) == 10
    assert pT.full_tree_id([0, 1, 1, 0]) == 13


def test_tree_generation():
    path = 'data/test_spectra/'
    for fname in os.listdir(path):
        with open(path+fname, 'rb') as f:
            data = pickle.load(f)
            tree = pT.tree_from_spectrum(data['spectrum'])
            for var in ['bounds', 'coords']:
                 assert [node[var] for node in data['tree'].values()] == [node[var] for node in tree.values()]


if __name__ == "__main__":

    #test_simple_tree()
    #test_complex_tree()
    test_tree_generation()