#!/bin/bash

files=(
    'output/20141005_1600_Cab_peakTree.nc4'
    'output/20141012_1421_Cab_peakTree.nc4'
    'output/20141012_1500_Cab_peakTree.nc4'
    'output/20141012_1559_Cab_peakTree.nc4'
    'output/20141012_1700_Cab_peakTree.nc4'
    'output/20141012_1800_Cab_peakTree.nc4'
    'output/20141016_0200_Cab_peakTree.nc4'
    'output/20141019_1400_Cab_peakTree.nc4'
    'output/20141019_1500_Cab_peakTree.nc4'
    'output/20141102_1400_Cab_peakTree.nc4'
    'output/20141102_1500_Cab_peakTree.nc4'
    'output/20141117_0000_Cab_peakTree.nc4'
    'output/20141117_0059_Cab_peakTree.nc4'
    'output/20141117_0200_Cab_peakTree.nc4'
    'output/20141117_2000_Cab_peakTree.nc4'
    'output/20141117_2100_Cab_peakTree.nc4'
    'output/20141117_2200_Cab_peakTree.nc4'
    'output/20141117_2300_Cab_peakTree.nc4'
    )

files=(
    'output/20141102_1600_Cab_peakTree.nc4'
    'output/20141102_1700_Cab_peakTree.nc4'
    'output/20141102_1800_Cab_peakTree.nc4'
    'output/20141102_1859_Cab_peakTree.nc4'
    )



for i in "${files[@]}"
do
    echo "$i"
    #ls "$i"
    #python3 plot2d.py "$i" --no-nodes 2
done

#python3 plot2d.py output/20170629_0830_Pol_peakTree.nc4 --range-interval 400-3243 --no-nodes 2

python3 convert_to_json.py output/20170629_0830_Pol_peakTree.nc4 20170629_0830_data --time-interval 0-450 --range-interval 0-100