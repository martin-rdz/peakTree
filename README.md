# peakTree



Technical documentation is available at [peakTree-doc](https://martin-rdz.github.io/peakTree-doc/)

### Requirements

peakTree requires python3 with following packages:
```
numpy==1.14.5
graphviz==0.8.2
matplotlib==2.2.2
netCDF4==1.4.2
```

### Setup

The peakTree software package should be included in a file structure similar to this example:
```
├── data                [input spectra]
├── docs                [code to generate the documentation using sphinx]
│   ├── Makefile
│   └── source
├── output              [converted data]
├── peakTree
│   ├── helpers.py
│   ├── __init__.py
│   ├── print_tree.py
│   ├── __pycache__
│   ├── test_peakTree.py
│   └── VIS_Colormaps.py
├── plot2d.py
├── plots               [standard folder for plots]
├── reader_example.py
├── README.md
├── requirements.txt
├── run_conversion.py
├── run_conversion.py.lprof
├── run_plots.sh
└── spectrum_example.py
```

Please update your meta information in the `output_meta.toml` file.


### License
Copyright 2018, Martin Radenz
[MIT License](<http://www.opensource.org/licenses/mit-license.php>)