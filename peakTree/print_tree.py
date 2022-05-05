#! /usr/bin/env python3
# coding=utf-8
"""
output of visualized trees as text, plots and schematic graphs
"""
"""
Author: radenz@tropos.de
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from . import helpers as h

import graphviz
import json


def coord_pattern_child(p):
    """the coordinate pattern required for filtering the dictionary for children
    for use in :func:`iterchilds`

    Args:
        p: coordinate of parent
    Returns:
        function that thest if a coordinate is a child of p
    """
    return lambda d: d['coords'][:-1] == p and len(d['coords']) == len(p)+1


def iterchilds(travtree, parentcoord):
    """generator that yields all childs of a parent with given coordinats
    for use in :func:`iternodes`
    """
    for n in list(filter(coord_pattern_child(parentcoord), travtree.values())):
        yield n
        yield from iterchilds(travtree, n['coords'])


def iternodes(travtree):
    """generator that yields a full traversal of the tree"""
    level_no=0
    nodes = list(filter(lambda d: len(d['coords']) == level_no+1, travtree.values()))
    for n in nodes:
        yield n
        yield from iterchilds(travtree, n['coords'])


def travtree2text(travtree, show_coordinats=True):
    """returns a string with the tabular representation of the traversed tree
    
    Args:
        travtree: traversed tree
        show_coordinates (optional): include the coordinates
    Returns:
        string with line breaks
    """
    lines = []
    levels = max(list(map(lambda v: len(v['coords']), travtree.values())))
    if show_coordinats:
        header = ' coordinates '+levels*'  '+ '                     Z       v    width    sk   LDR     t   LDRmax  prom'
    else:
        header = ' '+levels*'  '+ '             Z       v    width    sk   LDR     t   LDRmax  prom'
    lines.append(header)
    
    for v in iternodes(travtree):

        coords = '{:20s}'.format(str(v['coords']))
        bounds = '({:>3d}, {:>3d})'.format(*v['bounds'])
        sp_before = (len(v['coords'])-1)*'  '
        sp_after = (levels-len(v['coords']))*'  '

        mom1 = '{:> 6.2f}, {:> 6.2f}, {:>4.2f}'.format(h.lin2z(v['z']), v['v'], v['width'])
        mom2 = '{:> 3.2f}, {:> 5.1f}, {:> 5.1f}, {:> 5.1f}, {:> 5.1f}'.format(
            v['skew'], h.lin2z(v['ldr']), h.lin2z(v['thres']), h.lin2z(v['ldrmax']), h.lin2z(v['prominence']))
        #mom2 = '{:> 3.2f}, {:> 5.1f}, {:> 5.1f}'.format(v['skew'], h.lin2z(v['ldr']), h.lin2z(v['thres']))
        #txt = "{:>2d}{}{}{}{} {}\n{}{}".format(k, sp_before, bounds, sp_after, mmv, mom1, 33*' ', mom2)
        if show_coordinats:
            txt = "{}{} {}{} {}, {}".format(sp_before, bounds, coords, sp_after, mom1, mom2)
        else:
            txt = "{}{} {} {}, {}".format(sp_before, bounds, sp_after, mom1, mom2)
        lines.append(txt)
    return '\n'.join(lines)


def gen_lines_to_par(travtree):
    """get all the connection lines between the nodes of travtree
    
    Returns:
        list of coordinate pairs ``[(v,z), ..]``
    """
    chunks = []
    for k,v in travtree.items():
        if v['parent_id'] != -1:
            x = [v['v'], travtree[v['parent_id']]['v']]
            y = [v['z'], travtree[v['parent_id']]['z']]
            chunks.append((x,y))
    return chunks


def plot_spectrum(travtree, spectrum, savepath):
    """plot the spectrum together with the traversed tree
    
    Args:
        travtree: traversed tree
        spectrum: spectrum dict
        savepath: either ``None`` or string
    Returns:
        fig, ax
    """
    dt=h.ts_to_dt(spectrum['ts'])

    if 'decoupling' in spectrum:
        if 'specZco' in spectrum:
            decoupling_threshold = h.z2lin(
                h.lin2z(spectrum['specZco'])+spectrum['decoupling'])
        else:
            decoupling_threshold = h.z2lin(
                h.lin2z(spectrum['specZ'])+spectrum['decoupling'])
            spectrum['specZco'] = spectrum['specZ']
    # cut again to remove the smoothing effects at the edges
    specZ_cut = np.ma.masked_less(spectrum['specZ'], spectrum['noise_thres'])
    
    fig, ax = plt.subplots(1, figsize=(8, 7), sharex=True)

    #plot the tree structure
    for chunk in gen_lines_to_par(travtree):
        ax.plot(chunk[0], h.lin2z(np.array(chunk[1])), '-', color='grey')

    flatten = [(item[1]['v'],item[1]['z']) for item in travtree.items()]
    ax.plot([e[0] for e in flatten], h.lin2z(np.array([e[1] for e in flatten])), 'o', color='r', markersize=5)

    #ax.hlines(h.lin2z(valid_LDR), -10, 10, color='grey')
    ax.hlines(h.lin2z(spectrum['noise_thres']), -10, 10, ls='--', lw=1.3, color='grey')
    if 'noise_cx_thres' in spectrum:
        ax.hlines(h.lin2z(spectrum['noise_cx_thres']), -10, 10, 
                  color='plum', ls='--', lw=1.3)
    if 'specLDR' in spectrum:
        ax.step(spectrum['vel'], h.lin2z(spectrum['specLDR']), 
                linewidth=1.5, color='turquoise', where='mid', label='specLDR')
    if 'specLDRmasked' in spectrum:
        ax.step(spectrum['vel'], h.lin2z(spectrum['specLDRmasked']), 
                linewidth=1.5, color='blue', where='mid', label='specLDR')
    if 'specZ_raw' in spectrum:
        ax.step(spectrum['vel'], h.lin2z(spectrum['specZ_raw']), 
                linewidth=1.5, color='lightsalmon', where='mid', label='specZ raw')
    if 'specZco' in spectrum:
        ax.step(spectrum['vel'], h.lin2z(spectrum['specZco']), 
                linewidth=0.8, color='grey', where='mid')
        ax.step(spectrum['vel'], h.lin2z(decoupling_threshold), 
                linewidth=1.0, color='grey', where='mid', label='decoupling')

    ax.step(spectrum['vel'], h.lin2z(spectrum['specZ']),
            linewidth=1.5, color='pink', where='mid')
    ax.step(spectrum['vel'], h.lin2z(specZ_cut),
            linewidth=1.5, color='red', where='mid', label='specZ')
    if 'specZcx' in spectrum:
        ax.step(spectrum['vel'], h.lin2z(spectrum['specZcx']), 
                linewidth=1.5, color='darkviolet', where='mid', label='specZcx')
    ax.set_xlim([-6,3])
    ax.set_ylabel('Spectral Reflectivity [dBZ]')
    ax.set_xlabel('Velocity [m s$\\mathregular{^{-1}}$]')
    ax.set_ylim(bottom=-65)
    #special for the convective case
    # ax.set_ylim(bottom=-75)
    # ax.set_xlim([-8.5, 8.5])
    ax.set_ylim([-75, 2])

    ax.legend(loc='upper right')
    titlestr = '{} {:0>5.0f} m'.format(dt.strftime('%Y-%m-%d %H:%M:%S'), spectrum['range'])
    if 'ind_chirp' in spectrum:
        titlestr += f" chirp {spectrum['ind_chirp']}"
    ax.set_title(titlestr)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', width=1.5, right=True, top=True)
    ax.tick_params(axis='both', which='minor', width=1.3, right=True, top=True)
    fig.subplots_adjust(bottom=0.5, top=0.95)


    if travtree != {}:
        txt = travtree2text(travtree, show_coordinats=False)
        ax.text(0.03, 0.42, txt,
                horizontalalignment='left', verticalalignment='top',
                transform=fig.transFigure, fontsize=11, family='monospace',)

    if savepath is not None:
        savename = '{}_{:0>5.0f}m_spectrum.png'.format(dt.strftime('%Y-%m-%d_%H%M%S'), spectrum['range'])
        fig.savefig(savepath + savename, dpi=250)
    return fig, ax


def render_node_table(key, value):
    string = """{} [label=<<font face='helvetica' point-size="10"><table border="0" cellborder="0" cellspacing="0">
       <tr><td colspan='4'><font point-size='11'><B>node {}</B></font></td></tr>
       <tr><td>Z:</td><td align='right'>{:.1f}</td><td>  width:</td><td align='right'>{:.2f}</td></tr>
       <tr><td>v:</td><td align='right'>{:.2f}</td><td>  thres:</td><td align='right'>{:.1f}</td></tr>
     </table></font>>]""".format(key, key, h.lin2z(value['z']), value['width'], value['v'], h.lin2z(value['thres']))
    return string


def render_node_bounds(key, value):
    string = """{} [label=<<font face='helvetica' point-size="11">
    <table border="0" cellborder="0" cellspacing="0">
    <tr><td><B>node {}</B></td></tr>
    <tr><td>bin no {}-{}</td></tr></table></font>>]""".format(key, key, *value['bounds'])
    return string


def dot_format(travtree, display="table"):
    """generate a string for the graphviz dot format
    
    Args:
        travtree: traversed tree
        display: either ``table`` or ``bounds``
    Returns:
        a graphvis compatible definition string
    """
    print('dot format travtree', travtree)
    if display == "table":
        node_props = [render_node_table(*elem) for elem in travtree.items()]
        shape = "box"
    elif display == "bounds":
        node_props = [render_node_bounds(*elem) for elem in travtree.items()]
        shape = "box"
    connections = ['{} -> {};'.format(v['parent_id'], k) for k,v in travtree.items() if not v['parent_id'] == -1]
    string = ['digraph G { graph [fontname = "helvetica"] node [shape='+ shape +']'] + node_props + connections + ['}']
    # string = ['graph [fontname = "helvetica"] node [shape=ellipse] ',  
    #           'subgraph cluster1 { '+'label=<<font point-size="11">{} {:0>5.0f}m</font>>'.format(dt.strftime('%Y-%m-%d_%H%M%S'), rg)]\
    #            + node_props + connections + ['}']
    return '\n'.join(string)


def vis_tree(dot):
    """visualize the dot string"""
    src = graphviz.Source(dot)
    return src


def format_for_json(elem):
    """json elementwise formatter"""
    if isinstance(elem, np.integer):
        return int(elem)
    elif isinstance(elem, np.floating):
        return round(float(elem), 4)
    elif isinstance(elem, np.ndarray):
        return elem.tolist()
    elif isinstance(elem, float):
        return round(elem, 4)
    else:
        return elem


def d3_format(travtree):
    """format the traversed tree in a json compatible manner"""
    nodes = []
    for k, v in travtree.items():
        v['id'] = k
        v['bounds'] = list(map(int, v['bounds']))
        if v['parent_id'] == -1:
            del v['parent_id']
        v['z'] =h.lin2z(v['z']) 
        v['ldr'] = h.lin2z(v['ldr'])
        v['ldrmax'] = h.lin2z(v['ldrmax'])
        v['thres'] = h.lin2z(v['thres'])
        v = {ky: format_for_json(val) for ky, val in v.items()}
        nodes.append(v)
    return json.dumps(nodes)
        
        

