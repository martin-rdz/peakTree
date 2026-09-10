# %% [markdown]
# # MIRA E2E comparison
# 
# Aim is to compare znc processed peakTree with mmclx
# %%

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
files = sorted(list(Path('/data/level0/mira35/2023').glob('20231228*.znc')))

# %%
file = files[3]
file_ref = file.parent / file.with_suffix('.mmclx')
ds_input = peakTree.load_znc(file)

# %%
ds_input['noise_mask'] = ds_input['Z'] < ds_input['noise']*0.3
ds_input

# %%

# %%
# Experimental if substracting noise will help
# implemented in default loading function
#ds_input['Z'] = ds_input['Z'] - ds_input['noise']
#ds_input['Zcx'] = ds_input['Zcx'] - ds_input['noise_cx']

# %%

meta={
    'Z': ['M0', 'M1', 'M2', 'M3', 'P'],
    'Zcx': ['M0', 'P'],
    }


ds_rect = peakTree.ds_to_tree(
    ds_input,
    {'width_thres': 0.1, 'prom_thres': 1},
    meta
)

# %%

ds_rect['LDR'] = xr.where(
    ds_rect['Zcx_M0'] < 10**(-49/10), -999, ds_rect['Zcx_M0'] / ds_rect['Z_M0'])

# %%


with xr.open_dataset(file_ref) as ds_ref:
    ds_ref.load()

ds_ref['time'] = (ds_ref['time']*1e6 + ds_ref['microsec']).astype('datetime64[us]')


fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)

pcmesh1 = axes[0].pcolormesh(
    ds_rect['time'], ds_rect['range'], 10*np.log10(ds_rect['Z_M0'].isel(node=0)).T,
    shading='nearest',
    vmin=-45, vmax=10,
    cmap=cmc.roma_r
)
axes[0].set_title(f'Z node 0')
fig.colorbar(pcmesh1, ax=axes[0], label=f'Z')

pcmesh2 = axes[1].pcolormesh(
    ds_ref['time'], ds_ref['range'], 10*np.log10(ds_ref['Zg']).T,
    shading='nearest',
    vmin=-45, vmax=10,
    cmap=cmc.roma_r
)
axes[1].set_title('Ze mmclx')
fig.colorbar(pcmesh2, ax=axes[1], label='Ze node 0')

diff = 10*np.log10(ds_ref['Zg']) - 10*np.log10(ds_rect['Z_M0'].isel(node=0))
mean = diff.mean().values  # compute mean once before plotting
pcmesh3 = axes[2].pcolormesh(
    ds_ref['time'], ds_ref['range'], diff.T,
    shading='nearest',
    vmin=-5, vmax=5,
    cmap=cmc.vik
)
axes[2].set_title('Difference: mmclx Z - peakTree Z (node 0)')
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

fig.suptitle(file.name, fontsize=12, y=0.95)

fig.tight_layout(rect=[0, 0, 1, 0.97])  # adjust for suptitle if present

outname = f"{file.stem}_comparison_mmclx_pT_Z.png"
fig.savefig(outname, dpi=200)



# %%

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)

pcmesh1 = axes[0].pcolormesh(
    ds_rect['time'], ds_rect['range'], 10*np.log10(ds_rect['LDR'].isel(node=0)).T,
    shading='nearest',
    vmin=-30, vmax=0,
    cmap=cmc.roma_r
)
axes[0].set_title(f'LDR node 0')
fig.colorbar(pcmesh1, ax=axes[0], label=f'Z')

pcmesh2 = axes[1].pcolormesh(
    ds_ref['time'], ds_ref['range'], 10*np.log10(ds_ref['LDRg']).T,
    shading='nearest',
    vmin=-30, vmax=0,
    cmap=cmc.roma_r
)
axes[1].set_title('LDRg mmclx')
fig.colorbar(pcmesh2, ax=axes[1], label='Ze node 0')

diff = 10*np.log10(ds_ref['LDRg']) - 10*np.log10(ds_rect['LDR'].isel(node=0))
mean = diff.mean().values  # compute mean once before plotting
pcmesh3 = axes[2].pcolormesh(
    ds_ref['time'], ds_ref['range'], diff.T,
    shading='nearest',
    vmin=-5, vmax=5,
    cmap=cmc.vik
)
axes[2].set_title('Difference: mmclx LDRg - peakTree LDR (node 0)')
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

fig.suptitle(file.name, fontsize=12, y=0.95)

fig.tight_layout(rect=[0, 0, 1, 0.97])  # adjust for suptitle if present

outname = f"{file.stem}_comparison_mmclx_pT_LDR.png"
fig.savefig(outname, dpi=200)

# %%
