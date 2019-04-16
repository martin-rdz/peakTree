====================
Data formats
====================

Within peakTree basically two data formats are used to handle the trees.

Array
-----

The first one is array based and used to save the trees into netcdf4 files.
Such files are the output of the conversion from Doppler spectra to trees.
A variable exists for each moment ``Z``, ``v``, ``width`` , ... with a nodes
value saved according to its index. Hence, the array might be sparse.

.. code::

   Z(time, height, node)
   v(time, height, node)
   width(time, height, node)

To reconstruct the tree, the variable ``parent`` holds the index of a nodes parent
node. The :meth:`peakTree.peakTreeBuffer.get_tree_at` provieds a reading routine.

.. code:: python

    parent[1632,13,:] = [-1.0 0.0 0.0 -- -- 2.0 2.0]
    Z[1632,13,:] = [-5.2 -17.0 -5.4 -- -- -5.6 -19.5]


Dictionary
----------

The second one contains the tree as a dictionary in a 2d list (either as a numpy array with
dtype object or a javascript/json list of list with objects).
Each entry of the dictionary contains the moments of a node.
The trees are stored inside a dictionary under the key ``var`` with the additional keys
``ts``, ``rg``, ``dimlabel`` and ``paraminfo`` providing meta information.
This format is used for the interactive visualizer and grouping into particle populations.


.. code:: javascript

  [
    times... 
    [
      ranges...
      {0: {"id": 0, "bounds": [70, 164], "prominence": 6955.19, "skew": -0.2946, "v": -0.7237,
         "ldrmax": -13.8709, "width": 0.4588, "thres": -52.1102, "z": -5.2166, "ldr": -17.102, 
         "coords": [0]}, 
       1: {"id": 1, "bounds": [70, 110], "prominence": 3.1156, "skew": 0.3129, "v": -1.5912, 
         "ldrmax": -26.2039, "width": 0.099, "thres": -27.7299, "z": -17.0026, "ldr": -25.3611, 
         "coords": [0, 0], "parent_id": 0}, 
       2: {"id": 2, "bounds": [110, 164], "prominence": 25.3677, "skew": -0.0628, "v": -0.6729, 
         "ldrmax": -13.8709, "width": 0.3627, "thres": -27.7299, "z": -5.4885, "ldr": -16.8673, 
         "coords": [0, 1], "parent_id": 0}, 
       5: {"id": 5, "bounds": [110, 126], "prominence": 22.9612, "skew": -0.1676, "v": -0.6873, 
         "ldrmax": -13.8709, "width": 0.3429, "thres": -27.297, "z": -5.6327, "ldr": -16.7911, 
         "coords": [0, 1, 0], "parent_id": 2}, 
       6: {"id": 6, "bounds": [126, 164], "prominence": 1.3913, "skew": -0.1692, "v": 0.0501, 
         "ldrmax": -21.6985, "width": 0.0901, "thres": -27.297, "z": -19.5505, "ldr": -18.7583, 
         "coords": [0, 1, 1], "parent_id": 2}
      ...
    ]
    ...
  ]



