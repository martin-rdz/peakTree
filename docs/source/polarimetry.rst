
====================
Polarimetry
====================

Spectral polarimetric variables can be calculated for two configurations: ``STSR`` and ``LDR``.

.. figure:: _static/radar_pol_scheme.png
   :figwidth: 80 %
   :alt: Scheme of the polarimetric configurations

   Scheme of the polarimetric configurations. 



LDR mode
---------

With this configuration, the LDR can be calculated directly

.. math::
    
    \mathrm{LDR} = \frac{<E_\mathrm{x}^2>}{<E_\mathrm{c}^2>}

..
    describe filtering

LDR and LDRmax
----------------

The polarimetric properties are calculated as a peak average (with filtering of cross-channel noise and decoupling) as well as at the maximum of the co-channel reflectivity (ignoring cross-channel noise level and decoupling).

.. figure:: _static/spec_to_tree_one_example_ldr.png
   :figwidth: 70 %
   :alt: Illustration of LDR and LDRmax

   Illustration of LDR and LDRmax. 


STSR mode
----------

With the STSR configuration, the LDR cannot be observed directly, but has to be calculated from the correlation coefficient.
Following the assumption of isotropic scatterers [Galletti_Zrnic_2012]_ and reflection symmetry [Galletti_et_al_2011]_, as done in `rpgpy`_ ``calc_spectral_LDR()``:

.. math::

    \mathrm{SLDR} = 10 \log_{10} \left( \frac{1-\varrho_\mathrm{hv}}{1+\varrho_\mathrm{hv}} \right)


.. [Galletti_et_al_2011] Galletti, M., Zrnic, D.S., Melnikov, V.M., Doviak, R.J., 2011. Degree of polarization: theory and applications for weather radar at LDR mode, in: 2011 IEEE RadarCon (RADAR). Presented at the 2011 IEEE Radar Conference (RadarCon), IEEE, Kansas City, MO, USA, pp. 039–044. https://doi.org/10.1109/RADAR.2011.5960495

.. [Galletti_Zrnic_2012] Galletti, M., Zrnic, D.S., 2012. Degree of Polarization at Simultaneous Transmit: Theoretical Aspects. IEEE Geosci. Remote Sensing Lett. 9, 383–387. https://doi.org/10.1109/LGRS.2011.2170150

.. _rpgpy: https://github.com/actris-cloudnet/rpgpy/blob/master/rpgpy/spcutil.py