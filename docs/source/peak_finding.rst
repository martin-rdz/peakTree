======================
Peak finding
======================

The peak finding procedure is adapted to the peako [Kalesse_et_al_2019]_ parameters in v0.3.
Additional averaging in time and height can be set via ``t_avg`` and ``h_avg`` with the units of ``[s]`` and ``[m]``, respectively.
Smoothing along the velocity dimension can be done by 3 methods:

#. ``scipy.signal.savgol_filter`` by ``setting smooth_polyorder = 1`` (linear) or ``= 2`` (quadratic)
#. ``loess.loess_1d`` by ``setting smooth_polyorder = 11`` (linear) or ``= 12`` (quadratic)
#. convolution with a gauss function by ``setting smooth_polyorder = 21``

where the smoothing span is given by ``span`` in ``[m s-1]``.
The actual peak finding parameters are ``prom_thres`` in ``[dB]`` and ``width_thres`` in ``[m s-1]``.


Configuration
--------------
Below an example for a RPG FMCW with two chrips is shown.

.. code::

    [limrad_peako] 
        location = "Leipzig"
        shortname = "Lei"
        loader = 'rpgpy'
    [limrad_peako.settings] 
        grid_time = false
        max_no_nodes = 15 
        polarimetry = 'STSR' 
        smooth_cut_sequence = 'sc' # cs cut-smooth | sc smooth-cut
        tot_spec_scaling = 4
        decoupling = -30
        station_altitude = 22
    # chirps in zero-based indexing
    [limrad_peako.settings.peak_finding_params.chirp0]
        # new: seconds instead of index
        t_avg = 0            # s
        h_avg = 100          # m
        cal_offset = 0        # dB
        span = 0.15          # m s-1
        smooth_polyorder = 2  # 0 deactivate | 2 second order
        prom_thres = 0.5     # dB
        width_thres = 0      # m s-1
    [limrad_peako.settings.peak_finding_params.chirp1]
        t_avg = 15           # s
        h_avg = 0            # m
        cal_offset = 0        # dB
        span = 0.2           # m s-1
        smooth_polyorder = 2  # 0 deactivate | 2 second order
        prom_thres = 0.5     # dB
        width_thres = 0      # m s-1
        
        
    [Lacros_Pun]
        location = "Punta Arenas"
        shortname = "Pun"
        loader = "mira"
    [Lacros_Pun.settings]
        decoupling = -30
        grid_time = 6
        max_no_nodes = 15
        polarimetry = 'LDR'
        smooth_cut_sequence = 'sc' # cs cut-smooth | sc smooth-cut
        #roll_velocity = 65 # for 512 bins = 7.92 m/s
        roll_velocity = 98 # = 6.56 m/s
        station_altitude = 9
        #add_to_fname = '_rectwin'
    [Lacros_Pun.settings.peak_finding_params]
        thres_factor_co = 4.0 # supress spurious peaks introduced by tails
                              # back to 3 as the other tail filter should work better
        thres_factor_cx = 3.0
        cal_offset = 0        # dB
        tail_filter = true    # true | false | omit; only tested with LACROS-Mira 
        # new: seconds instead of index
        t_avg = 0             # s
        h_avg = 0             # m
        span = 0.3            # m s-1
        smooth_polyorder = 2  # 0 deactivate | 2 second order
        prom_thres = 0.5      # dB
        width_thres = 0.05    # m s-1




.. [Kalesse_et_al_2019] Kalesse, H., Vogl, T., Paduraru, C., Luke, E., 2019. Development and validation of a supervised machine learning radar Doppler spectra peak-finding algorithm. Atmos. Meas. Tech. 12, 4591–4617. https://doi.org/10.5194/amt-12-4591-2019
