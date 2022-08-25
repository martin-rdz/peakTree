=============================
Tutorial
=============================

.. note::

    The following material is assembled for the ERAD22 Academy short course 'Polarimentric microphysical fingerprints in observations and NWP'


Setup
-------

.. code:: bash

	python3 -m venv peakTree-env
	source peakTree-env/bin/activate
	python3 -m pip install Cython
	python3 -m pip install pyLARDA
	python3 -m pip install rpgpy
	python3 -m pip install jupyter graphviz loess

	# either get a fresh clone of peakTree
	git clone https://github.com/martin-rdz/peakTree.git
	# or pull the latest update
	git pull origin master

	cd peakTree/tutorials
	# start the jupyter notebook (and your default browser should fire up)
	jupyter notebook
	# better use the following command
	# to ensure the correct executable is used
	../../peakTree-env/bin/jupyter notebook



Jupyter Notebooks
-------------------

01_peak_finding_tree_generation.ipynb
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a first step, we are looking at single spectra from a Metek MIRA-35 and how different peak finding parameters affect the generated binary tree.

02_convert_file.ipynb
^^^^^^^^^^^^^^^^^^^^^^^^^^

A full file of spectra (i.e. chunk of 1h) can also be converted conveniently to the peakTree output format.

03_analyze_output.ipynb
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a final step, let's have a look how to analyze an interpret a time-height slice of multipeak data.

Hints
-------------------

01_peak_finding_tree_generation.ipynb
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

	pTB = peakTree.peakTreeBuffer(config_file='../instrument_config.toml', system='limrad_punta')
	pTB.load('../data/190822_080000_P05_ZEN.LV0', load_to_ram=True)

02_convert_file.ipynb
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

	# it might be necessary to run
	!python3 -m pip install rpgpy
	# first in it's own cell and restart the kernel

	pTB = peakTree.peakTreeBuffer(config_file='../instrument_config.toml', system='limrad_punta')
	pTB.load_spec_file('../data/190822_070001_P05_ZEN.LV0', load_to_ram=True)
	pTB.assemble_time_height('../output/')