======================
Peak finding
======================

The peak finding procedure is adapted to the peako [Kalesse_et_al_2019]_ parameters in v0.3.
Most importantly, this means switching to LOESS smoothing and a generic peak-finding function.


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
        tot_spec_scaling = 4
        decoupling = -30
        station_altitude = 22
    # chirps in zero-based indexing
    [limrad_peako.settings.peak_finding_params.chirp0]
        # new: seconds instead of index
        t_avg = 0            # s
        h_avg = 100          # m
        span = 0.15          # m s-1
        smooth_polyorder = 1
        prom_thres = 0.5     # dB
        width_thres = 0      # m s-1
    [limrad_peako.settings.peak_finding_params.chirp1]
        t_avg = 15           # s
        h_avg = 0            # m
        span = 0.2           # m s-1
        smooth_polyorder = 1
        prom_thres = 0.5     # dB
        width_thres = 0      # m s-1



.. [Kalesse_et_al_2019] Kalesse, H., Vogl, T., Paduraru, C., Luke, E., 2019. Development and validation of a supervised machine learning radar Doppler spectra peak-finding algorithm. Atmos. Meas. Tech. 12, 4591–4617. https://doi.org/10.5194/amt-12-4591-2019
