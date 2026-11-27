import numpy as np
import pytest
import xarray as xr

import peakTree as pT
import peakTree.helpers as h


def synthetic(nave=20, seed=1):
    """strong (40 dB) + weak (12 dB) peak, averaged-exponential noise, noise subtracted"""
    rng = np.random.default_rng(seed)
    v = np.linspace(-10.5, 10.5, 512, endpoint=False)
    noise = 1.0
    true = 1e4 * np.exp(-0.5 * (v / 0.15)**2) + 10**1.2 * np.exp(-0.5 * ((v + 1.2) / 0.08)**2)
    raw = (true + noise) * rng.gamma(nave, 1 / nave, size=(2, 3, v.size))
    Z = (raw - noise).astype('f4')
    Z[Z < 1e-8] = 0
    da = xr.DataArray(Z, dims=('time', 'range', 'doppler'), coords={'doppler': v})
    nn = xr.DataArray(np.full((2, 3), noise, 'f4'), dims=('time', 'range'))
    return da, nn, v, true


@pytest.mark.parametrize('kernel', ['box', 'triangle', 'gauss', 'savgol'])
def test_doppler_kernel(kernel):
    w = h.doppler_kernel(7, kernel)
    assert w.shape == (7,)
    assert np.isclose(w.sum(), 1)
    assert np.allclose(w, w[::-1])


def test_doppler_kernel_even_window():
    with pytest.raises(ValueError):
        h.doppler_kernel(4)


def test_lin_box_is_periodic_moving_mean():
    Z, N, _, _ = synthetic()
    s = pT.smooth_doppler(Z, N, window=3, kernel='box', domain='lin').values
    ref = (np.roll(Z.values, 1, -1) + Z.values + np.roll(Z.values, -1, -1)) / 3
    ref[ref < 1e-8] = 0
    np.testing.assert_allclose(s, ref, rtol=1e-5)


@pytest.mark.parametrize('domain', ['lin', 'log'])
def test_roll_invariance(domain):
    Z, N, _, _ = synthetic()
    s = pT.smooth_doppler(Z, N, domain=domain).values
    s_rolled = pT.smooth_doppler(Z.roll(doppler=97), N, domain=domain).values
    np.testing.assert_allclose(np.roll(s, 97, -1), s_rolled, rtol=1e-5)


def test_log_noise_only_stays_zero():
    Z, N, _, _ = synthetic()
    Z = xr.zeros_like(Z)
    s = pT.smooth_doppler(Z, N, kernel='savgol', window=7).values
    assert (s == 0).all()


def test_log_preserves_flanks_and_no_negatives():
    Z, N, v, true = synthetic()
    s_lin = pT.smooth_doppler(Z, N, window=5, kernel='box', domain='lin').values[0, 0]
    s_log = pT.smooth_doppler(Z, N, window=5, kernel='gauss', domain='log').values[0, 0]
    s_sg = pT.smooth_doppler(Z, N, window=7, kernel='savgol', domain='log').values
    assert (s_sg >= 0).all()
    # flank of the strong peak (true 0..10 dB): log is closer to the truth than lin
    flank = (true > 1) & (true < 10) & (np.abs(v) < 1)
    err_lin = np.abs(10 * np.log10(s_lin[flank] / true[flank])).mean()
    err_log = np.abs(10 * np.log10(s_log[flank] / true[flank])).mean()
    assert err_log < err_lin
    # weak peak survives
    i = np.argmin(np.abs(v + 1.2))
    assert 10 * np.log10(s_log[i]) > 8


def test_dask_matches_numpy():
    Z, N, _, _ = synthetic()
    s = pT.smooth_doppler(Z, N).values
    s_dask = pT.smooth_doppler(Z.chunk(time=1), N.chunk(time=1))
    assert s_dask.chunks is not None
    np.testing.assert_allclose(s, s_dask.compute().values)
    assert s.dtype == np.float32
