#! /usr/bin/env python3
# coding=utf-8

import datetime
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

import argparse
import sys, os
from copy import copy
import peakTree
import peakTree.helpers as h
import peakTree.VIS_Colormaps as VIS_Colormaps

parser = argparse.ArgumentParser(description='Plot the peakTree converted files')
parser.add_argument('file', help='path of the file')
parser.add_argument('--no-nodes', type=int, default=0, help='plot nodes up to this no (including)')
parser.add_argument('--range-interval', type=str, default='min,10000', help='range to plot. e.g. min,7000, min,max, 500,max')
parser.add_argument('--time-interval', type=str, default='', help='range to plot. e.g. min,7000, min,max, 500,max')
parser.add_argument('--plotsubfolder', default='', help='path of the plot subfolder')
parser.add_argument('--system', default='', help='system identifier')
args = parser.parse_args()

pTB = peakTree.peakTreeBuffer(system=args.system)
pTB.load_peakTree_file(args.file)

if args.plotsubfolder == '':
    savepath = 'plots/'
else:
    savepath = 'plots/{}/'.format(args.plotsubfolder)
if not os.path.isdir(savepath):
    os.makedirs(savepath)

dt_list = [h.ts_to_dt(ts) for ts in pTB.timestamps]
print("temp_gradient ", np.percentile(np.gradient(pTB.timestamps), [10,25,50,75,90]))
hrange = pTB.range
no_nodes = pTB.f.variables['no_nodes'][:]

plot_nodes = list(range(args.no_nodes + 1))
plot_height_interval = args.range_interval.split(',')
if plot_height_interval[0] == "min":
    plot_height_interval[0] = hrange[0]
if plot_height_interval[1] == "max":
    plot_height_interval[1] = hrange[-1]
plot_height_interval = list(map(float, plot_height_interval))

if args.time_interval != '':
    print("time interval given")
    plot_time_interval = args.time_interval.split(',')
    if plot_time_interval[0] == "min":
        plot_time_interval[0] = dt_list[0]
    else:
        plot_time_interval[0] = datetime.datetime.strptime(plot_time_interval[0], "%Y%m%d_%H%M")
    if plot_time_interval[1] == "max":
        plot_time_interval[1] = dt_list[-1]
    else:
        plot_time_interval[1] = datetime.datetime.strptime(plot_time_interval[1], "%Y%m%d_%H%M")
else:
    plot_time_interval = [dt_list[0], dt_list[-1]]
print(plot_time_interval)

# use a categorial colormap for the no of nodes
#cat_cmap = plt.cm.get_cmap('terrain_r', 6)
cat_cmap = matplotlib.colors.ListedColormap(
    ["#ffffff", "#cdbfbc", "#987b61", "#fdff99", "#35d771", "#1177dd"], 'terrain_seq')
cat_cmap = matplotlib.colors.ListedColormap(
    #["#ffffff", "#cdbfbc", "orangered", "#fdff99", "#35d771", "#1177dd"], 'terrain_seq')
    #["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#999933", "#332288"], 'pTcat')
    ["#ffffff", "#cccccc", "#cc6677", "#88ccee", "#eecc66", "#332288"], 'pTcat') # yellow
    #based on paul tol vibrant
    #["#ffffff", "#cccccc", "#ee3377", "#33bbee", "#ee7733", "#0077bb"], 'pTcat') 
# based on paul tol (https://personal.sron.nl/~pault/)
#cat_cmap = matplotlib.colors.ListedColormap(
#    ["#ffffff", "#cccccc", "#cc6677", "#117733", "#33bbee", "#332288"], 'test')
# We must be sure to specify the ticks matching our target names
labels = {0: '0', 1: "1", 2: "3", 3: "5", 4: "7", 5: "9"}
# number of peaks instead number of nodes
labels = {0: '0', 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
cbarformatter = plt.FuncFormatter(lambda val, loc: labels[val])
no_nodes_plot = np.ceil(np.array(no_nodes)/2.)

txt = f"{pTB.f.getncattr('creation_time')} with {pTB.f.getncattr('commit_id')}"

fig, ax = plt.subplots(1, figsize=(10, 5.7))
pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list),
                       hrange,
                       np.transpose(no_nodes_plot),
                       cmap=cat_cmap, vmin=-0.5, vmax=5.5)
#cbar = fig.colorbar(pcmesh)
cbar = fig.colorbar(pcmesh, ticks=[0, 1, 2, 3, 4, 5], format=cbarformatter)
#ax.set_xlim([dt_list[0], dt_list[-1]])
#ax.set_ylim([height_list[0], height_list[-1]])
ax.set_xlim(plot_time_interval)
#ax.set_ylim([hrange[0], hrange[-1]])
ax.set_ylim(plot_height_interval)
ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
#cbar.ax.set_ylabel("Number of nodes", fontweight='semibold', fontsize=15)
cbar.ax.set_ylabel("Number of peaks", fontweight='semibold', fontsize=15)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

# ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
# ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
# ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,5)))
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

ax.tick_params(axis='both', which='both', right=True, top=True)
ax.tick_params(axis='both', which='major', labelsize=14, 
               width=3, length=5.5)
ax.tick_params(axis='both', which='minor', width=2, length=3)
cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                    width=2, length=4)
ax.text(1.15, -0.12, txt,
        fontsize=8, horizontalalignment='right',
        transform=ax.transAxes)

savename = savepath + dt_list[0].strftime("%Y%m%d_%H%M") + "_no_nodes.png"
fig.savefig(savename, dpi=250)


thres_0 = pTB.f.variables['threshold'][:,:,0]

fig, ax = plt.subplots(1, figsize=(10, 5.7))
pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list),
                    hrange,
                    np.transpose(thres_0),
                    cmap='jet', vmin=-70, vmax=-30)
cbar = fig.colorbar(pcmesh)
#ax.set_xlim([dt_list[0], dt_list[-1]])
#ax.set_ylim([height_list[0], height_list[-1]])
ax.set_xlim(plot_time_interval)
#ax.set_ylim([hrange[0], hrange[-1]])
#ax.set_ylim([hrange[0], 10000])
ax.set_ylim(plot_height_interval)
ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
cbar.ax.set_ylabel("Threshold node 0 [dBZ]", fontweight='semibold', fontsize=15)
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

# ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
# ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
# ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,5)))
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

ax.tick_params(axis='both', which='both', right=True, top=True)
ax.tick_params(axis='both', which='major', labelsize=14, 
            width=3, length=5.5)
ax.tick_params(axis='both', which='minor', width=2, length=3)
cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                    width=2, length=4)
ax.text(1.15, -0.12, txt,
        fontsize=8, horizontalalignment='right',
        transform=ax.transAxes)

savename = savepath + "{}_threshold_node0.png".format(dt_list[0].strftime("%Y%m%d_%H%M"))
fig.savefig(savename, dpi=250)

for i in plot_nodes:
    print('plot node ', i)
    Z_node = pTB.f.variables['Z'][:,:,i]

    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list),
                        hrange,
                        np.transpose(Z_node),
                        cmap='jet', vmin=-45, vmax=10)
    cbar = fig.colorbar(pcmesh)
    #ax.set_xlim([dt_list[0], dt_list[-1]])
    #ax.set_ylim([height_list[0], height_list[-1]])
    ax.set_xlim(plot_time_interval)
    #ax.set_ylim([hrange[0], hrange[-1]])
    #ax.set_ylim([hrange[0], 10000])
    ax.set_ylim(plot_height_interval)
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Reflectivity [dBZ]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

    # ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
    # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
    # ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
    ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,5)))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=14, 
                width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)
    ax.text(1.15, -0.12, txt,
            fontsize=8, horizontalalignment='right',
            transform=ax.transAxes)

    savename = savepath + "{}_reflectivity_node{}.png".format(dt_list[0].strftime("%Y%m%d_%H%M"), i)
    fig.savefig(savename, dpi=250)


    v_node = pTB.f.variables['v'][:,:,i]
    fig, ax = plt.subplots(1, figsize=(10, 5.7))
    pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list),
                        hrange,
                        np.transpose(v_node),
                        cmap=VIS_Colormaps.carbonne_map, vmin=-3, vmax=3)
    cbar = fig.colorbar(pcmesh)
    #ax.set_xlim([dt_list[0], dt_list[-1]])
    #ax.set_ylim([height_list[0], height_list[-1]])
    ax.set_xlim(plot_time_interval)
    #ax.set_ylim([hrange[0], hrange[-1]])
    #ax.set_ylim([hrange[0], 10000])
    ax.set_ylim(plot_height_interval)
    ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
    ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
    cbar.ax.set_ylabel("Velocity [m s$^\mathrm{-1}$]", fontweight='semibold', fontsize=15)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

    # ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
    # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
    # ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
    ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
    ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,5)))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    ax.tick_params(axis='both', which='both', right=True, top=True)
    ax.tick_params(axis='both', which='major', labelsize=14, 
                width=3, length=5.5)
    ax.tick_params(axis='both', which='minor', width=2, length=3)
    cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                        width=2, length=4)
    ax.text(1.15, -0.12, txt,
            fontsize=8, horizontalalignment='right',
            transform=ax.transAxes)

    savename = savepath + "{}_velocity_node{}.png".format(dt_list[0].strftime("%Y%m%d_%H%M"), i)
    fig.savefig(savename, dpi=250)

    if 'ldrmax' in pTB.f.variables:
        ldrmax_node = pTB.f.variables['ldrmax'][:,:,i]
        cmap = VIS_Colormaps.ldr_map
        cmap = 'jet'
        cmap = copy(matplotlib.cm.get_cmap('jet'))
        cmap.set_under('grey', 1.0)
        fig, ax = plt.subplots(1, figsize=(10, 5.7))
        pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list),
                            hrange,
                            np.transpose(ldrmax_node),
                            cmap=cmap, vmin=-32, vmax=-7)
        cbar = fig.colorbar(pcmesh, extend='min')
        #ax.set_xlim([dt_list[0], dt_list[-1]])
        #ax.set_ylim([height_list[0], height_list[-1]])
        ax.set_xlim(plot_time_interval)
        #ax.set_ylim([hrange[0], hrange[-1]])
        #ax.set_ylim([hrange[0], 10000])
        ax.set_ylim(plot_height_interval)
        ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
        ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
        cbar.ax.set_ylabel("LDRmax [dB]", fontweight='semibold', fontsize=15)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

        # ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
        # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
        # ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,5)))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        ax.tick_params(axis='both', which='both', right=True, top=True)
        ax.tick_params(axis='both', which='major', labelsize=14, 
                    width=3, length=5.5)
        ax.tick_params(axis='both', which='minor', width=2, length=3)
        cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                            width=2, length=4)
        ax.text(1.15, -0.12, txt,
               fontsize=8, horizontalalignment='right',
               transform=ax.transAxes)

        savename = savepath + "{}_ldrmax_node{}.png".format(dt_list[0].strftime("%Y%m%d_%H%M"), i)
        fig.savefig(savename, dpi=250)

    if "LDR" in pTB.f.variables:
        ldr_node = pTB.f.variables['LDR'][:,:,i]
        cmap = VIS_Colormaps.ldr_map
        cmap = 'jet'
        cmap = copy(matplotlib.cm.get_cmap('jet'))
        cmap.set_under('silver', 1.0)
        fig, ax = plt.subplots(1, figsize=(10, 5.7))
        pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list),
                            hrange,
                            np.transpose(ldr_node),
                            cmap=cmap, vmin=-32, vmax=-7)
        cbar = fig.colorbar(pcmesh, extend='min')
        #ax.set_xlim([dt_list[0], dt_list[-1]])
        #ax.set_ylim([height_list[0], height_list[-1]])
        ax.set_xlim(plot_time_interval)
        #ax.set_ylim([hrange[0], hrange[-1]])
        #ax.set_ylim([hrange[0], 10000])
        ax.set_ylim(plot_height_interval)
        ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
        ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
        cbar.ax.set_ylabel("LDR [dB]", fontweight='semibold', fontsize=15)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

        # ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
        # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
        # ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,5)))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        ax.tick_params(axis='both', which='both', right=True, top=True)
        ax.tick_params(axis='both', which='major', labelsize=14, 
                    width=3, length=5.5)
        ax.tick_params(axis='both', which='minor', width=2, length=3)
        cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                            width=2, length=4)
        ax.text(1.15, -0.12, txt,
               fontsize=8, horizontalalignment='right',
               transform=ax.transAxes)

        savename = savepath + "{}_ldr_node{}.png".format(dt_list[0].strftime("%Y%m%d_%H%M"), i)
        fig.savefig(savename, dpi=250)

    if 'ldrmin' in pTB.f.variables:
        ldrmin_node = pTB.f.variables['ldrmin'][:,:,i]
        cmap = copy(matplotlib.cm.get_cmap('jet'))
        cmap.set_under('grey', 1.0)
        fig, ax = plt.subplots(1, figsize=(10, 5.7))
        pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list),
                            hrange,
                            np.transpose(ldrmin_node),
                            cmap=cmap, vmin=-32, vmax=-7)
        cbar = fig.colorbar(pcmesh, extend='min')
        #ax.set_xlim([dt_list[0], dt_list[-1]])
        #ax.set_ylim([height_list[0], height_list[-1]])
        ax.set_xlim(plot_time_interval)
        #ax.set_ylim([hrange[0], hrange[-1]])
        #ax.set_ylim([hrange[0], 10000])
        ax.set_ylim(plot_height_interval)
        ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
        ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
        cbar.ax.set_ylabel("LDRmin [dB]", fontweight='semibold', fontsize=15)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

        # ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
        # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
        # ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,5)))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        ax.tick_params(axis='both', which='both', right=True, top=True)
        ax.tick_params(axis='both', which='major', labelsize=14, 
                    width=3, length=5.5)
        ax.tick_params(axis='both', which='minor', width=2, length=3)
        cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                            width=2, length=4)
        ax.text(1.15, -0.12, txt,
               fontsize=8, horizontalalignment='right',
               transform=ax.transAxes)

        savename = savepath + "{}_ldrmin_node{}.png".format(dt_list[0].strftime("%Y%m%d_%H%M"), i)
        fig.savefig(savename, dpi=250)

    if 'ldrmin' in pTB.f.variables:
        diff_1 = pTB.f.variables['ldrmax'][:,:,i] - pTB.f.variables['ldrmin'][:,:,i]
        cmap = 'RdBu'
        fig, ax = plt.subplots(1, figsize=(10, 5.7))
        pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list),
                            hrange,
                            np.transpose(diff_1),
                            cmap=cmap, vmin=-7, vmax=7)
        cbar = fig.colorbar(pcmesh)
        #ax.set_xlim([dt_list[0], dt_list[-1]])
        #ax.set_ylim([height_list[0], height_list[-1]])
        ax.set_xlim(plot_time_interval)
        #ax.set_ylim([hrange[0], hrange[-1]])
        #ax.set_ylim([hrange[0], 10000])
        ax.set_ylim(plot_height_interval)
        ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
        ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
        cbar.ax.set_ylabel("LDRmax - LDRmin [dB]", fontweight='semibold', fontsize=15)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

        # ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
        # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
        # ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,5)))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        ax.tick_params(axis='both', which='both', right=True, top=True)
        ax.tick_params(axis='both', which='major', labelsize=14, 
                    width=3, length=5.5)
        ax.tick_params(axis='both', which='minor', width=2, length=3)
        cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                            width=2, length=4)
        ax.text(1.15, -0.12, txt,
               fontsize=8, horizontalalignment='right',
               transform=ax.transAxes)

        savename = savepath + "{}_ldrmax-min_node{}.png".format(dt_list[0].strftime("%Y%m%d_%H%M"), i)
        fig.savefig(savename, dpi=250)

    if 'ldrleft' in pTB.f.variables:
        diff_1 = pTB.f.variables['ldrright'][:,:,i] - pTB.f.variables['ldrleft'][:,:,i]
        cmap = 'RdBu'
        fig, ax = plt.subplots(1, figsize=(10, 5.7))
        pcmesh = ax.pcolormesh(matplotlib.dates.date2num(dt_list),
                            hrange,
                            np.transpose(diff_1),
                            cmap=cmap, vmin=-7, vmax=7)
        cbar = fig.colorbar(pcmesh)
        #ax.set_xlim([dt_list[0], dt_list[-1]])
        #ax.set_ylim([height_list[0], height_list[-1]])
        ax.set_xlim(plot_time_interval)
        #ax.set_ylim([hrange[0], hrange[-1]])
        #ax.set_ylim([hrange[0], 10000])
        ax.set_ylim(plot_height_interval)
        ax.set_xlabel("Time UTC", fontweight='semibold', fontsize=15)
        ax.set_ylabel("Height", fontweight='semibold', fontsize=15)
        cbar.ax.set_ylabel("LDRright - LDRleft [dB]", fontweight='semibold', fontsize=15)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))

        # ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=3))
        # ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
        # ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=[0,20,40,60]))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0,61,5)))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

        ax.tick_params(axis='both', which='both', right=True, top=True)
        ax.tick_params(axis='both', which='major', labelsize=14, 
                    width=3, length=5.5)
        ax.tick_params(axis='both', which='minor', width=2, length=3)
        cbar.ax.tick_params(axis='both', which='major', labelsize=14,
                            width=2, length=4)
        ax.text(1.15, -0.12, txt,
               fontsize=8, horizontalalignment='right',
               transform=ax.transAxes)

        savename = savepath + "{}_ldrright-left_node{}.png".format(dt_list[0].strftime("%Y%m%d_%H%M"), i)
        fig.savefig(savename, dpi=250)