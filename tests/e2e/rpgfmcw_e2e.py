# %% [markdown]
# # RPGFMCW E2E comparison
# 
# Aim is to compare lv0 (binary) processed peakTree with lv1 (binary/netcdf?)

# %%
import datetime 

import xarray as xr

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import cmcrameri.cm as cmc

import sys
sys.path.append('../../../peakTree/')
import peakTree

# %%
filename = Path("/data/userdata/miratools/case_cloudlab/220220_020002_P02_ZEN.LV0")

file_ref = filename.parent / filename.with_suffix('.LV1.NC')
print(file_ref)

# %%
dt_input = peakTree.load_rpgbinary(filename)

# %%

def noise_thresholding(ds, config): 
    if not ds:
        return None
    ds = ds.copy()
    ds['noise_mask'] = ds['Z'] < ds['noise']*config['Q']
    return ds

dt_input = peakTree.map_over_dataset_nested_args(dt_input, noise_thresholding, {'Q': 0.3})

# %%

meta={
    'Z': ['M0', 'M1', 'M2', 'M3', 'P'],
    'Zcx': ['M0', 'P'],
    }


dt_rect = peakTree.to_tree(
    dt_input,
    {'width_thres': 0.1, 'prom_thres': 1},
    meta
)

# %%
combined_ds = xr.concat([dt_rect[child].to_dataset() for child in dt_rect.keys()], dim='range')

# %%
combined_ds['LDR'] = xr.where(
    combined_ds['Zcx_M0'] < 10**(-49/10), -999, combined_ds['Zcx_M0'] / combined_ds['Z_M0'])

# %%

with xr.open_dataset(file_ref) as ds_ref:
    ds_ref.load()

offset = (datetime.datetime(2001,1,1) - datetime.datetime(1970, 1, 1)).total_seconds()
ds_ref['Time'] = (offset*1e3 + ds_ref['Time']*1e3 + ds_ref['Timems']).astype('datetime64[ms]')
#ds_ref['time'] = (ds_ref['time']*1e6 + ds_ref['microsec']).astype('datetime64[us]')
#with xr.set_options(display_max_rows=200):
#    print(ds_ref)

ds_ref = ds_ref.rename({
    "Time": "time",
})

ds_ref

# %%

ze = xr.concat(
    [
        ds_ref["C1ZE"].rename({"C1Range": "Range"}),
        ds_ref["C2ZE"].rename({"C2Range": "Range"}),
        ds_ref["C3ZE"].rename({"C3Range": "Range"}),
    ],
    dim="Range",
)

ze = ze.rename({
    "Range": "range",
})

sldr = xr.concat(
    [
        ds_ref["C1SLDR"].rename({"C1Range": "Range"}),
        ds_ref["C2SLDR"].rename({"C2Range": "Range"}),
        ds_ref["C3SLDR"].rename({"C3Range": "Range"}),
    ],
    dim="Range",
)

sldr = sldr.rename({
    "Range": "range",
})
sldr = sldr.where(sldr >= -99)



fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)

pcmesh1 = axes[0].pcolormesh(
    combined_ds['time'], combined_ds['range'], 10*np.log10(combined_ds['Z_M0'].isel(node=0)).T,
    shading='nearest',
    vmin=-45, vmax=10,
    cmap=cmc.roma_r
)
axes[0].set_title(f'Z node 0')
fig.colorbar(pcmesh1, ax=axes[0], label=f'Z')

pcmesh2 = axes[1].pcolormesh(
    ze['time'], ze['range'], 10*np.log10(ze).T,
    shading='nearest',
    vmin=-45, vmax=10,
    cmap=cmc.roma_r
)
axes[1].set_title('Ze lv1')
fig.colorbar(pcmesh2, ax=axes[1], label='Ze node 0')

diff = 10*np.log10(ze) - 10*np.log10(combined_ds['Z_M0'].isel(node=0))
mean = diff.mean().values  # compute mean once before plotting
pcmesh3 = axes[2].pcolormesh(
    ze['time'], ze['range'], diff.T,
    shading='nearest',
    vmin=-5, vmax=5,
    cmap=cmc.vik
)
axes[2].set_title('Difference: lv1 Z - peakTree Z (node 0)')
fig.colorbar(pcmesh3, ax=axes[2], label='Difference')

# Shared x-axis formatting
for ax in axes:
    ax.set_ylabel('Range [m]')
    ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    ax.set_ylim(0, 6000)

# Annotate mean on the bottom subplot
fig.text(0.99, 0.95, f"mean {mean:.3f}", 
         horizontalalignment='right', transform=axes[2].transAxes)

fig.suptitle(filename.name, fontsize=12, y=0.95)

fig.tight_layout(rect=[0, 0, 1, 0.97])  # adjust for suptitle if present

outname = f"{filename.stem}_comparison_lv1_pT_Z.png"
fig.savefig(outname, dpi=200)

# %%

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)

# Top subplot: LDR from combined_ds (node 0)
pcmesh1 = axes[0].pcolormesh(
    combined_ds['time'], combined_ds['range'], 10*np.log10(combined_ds['LDR'].isel(node=0)).T,
    shading='nearest',
    vmin=-30, vmax=0,
    cmap=cmc.roma_r
)
axes[0].set_title(f'LDR node 0')
fig.colorbar(pcmesh1, ax=axes[0], label=f'LDR node {0}')

# Middle subplot: SLDR from lv1.nc
pcmesh2 = axes[1].pcolormesh(
    sldr['time'], sldr['range'], sldr.T,
    shading='nearest',
    vmin=-30, vmax=0,
    cmap=cmc.roma_r
)
axes[1].set_title('SLDR lv1')
fig.colorbar(pcmesh2, ax=axes[1], label='SLDR')

# Bottom subplot: Difference (SLDR - LDR)
diff = sldr - 10*np.log10(combined_ds['LDR'].isel(node=0))
mean = diff.mean().values  # compute mean once before plotting
pcmesh3 = axes[2].pcolormesh(
    sldr['time'], sldr['range'], diff.T,
    shading='nearest',
    vmin=-5, vmax=5,
    cmap=cmc.vik
)
axes[2].set_title('Difference: lv1 SLDR - peakTree LDR (node 0)')
fig.colorbar(pcmesh3, ax=axes[2], label='Difference')

# Shared x-axis formatting
for ax in axes:
    ax.set_ylabel('Range [m]')
    ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
    ax.set_ylim(0, 6000)

# Annotate mean on the bottom subplot
fig.text(0.99, 0.95, f"mean {mean:.3f}", 
         horizontalalignment='right', transform=axes[2].transAxes)

fig.suptitle(filename.name, fontsize=12, y=0.95)

fig.tight_layout(rect=[0, 0, 1, 0.97])  # adjust for suptitle if present
outname = f"{filename.stem}_comparison_lv1_pT_SLDR.png"
fig.savefig(outname, dpi=200)



