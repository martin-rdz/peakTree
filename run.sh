#!/bin/bash


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
    python3 plot2d.py "$i" --no-nodes 2
done

python3 plot2d.py output/20170629_0830_Pol_peakTree.nc4 --range-interval 400,3243 --no-nodes 2

#python3 convert_to_json.py output/20170629_0830_Pol_peakTree.nc4 output/20170629_0830_data --time-interval 0-450 --range-interval 0-100
#python3 convert_to_json.py output/20170629_0830_Pol_peakTree.nc4 output/20170629_0830_data --time-interval 0-50 --range-interval 0-100


#python3 plot2d.py output/20140221_2232_kazrbaeccpeako_peakTree.nc4 --no-nodes 2 --range-interval 1800,6500 --time-interval 20140221_2230,20140221_2247
python3 plot2d.py output/20140221_2200_kazrbaecc_peakTree.nc4 --no-nodes 2 --range-interval 1800,6500 --time-interval 20140221_2230,20140221_2247
#python3 plot2d.py output/20140221_2200_kazrbaecc_peakTree.nc4 --no-nodes 2

# plot second case
#python3 plot2d.py output/20140202_1600_kazrbaeccpeako_peakTree.nc4 --no-nodes 6 --range-interval 400,3000
python3 plot2d.py output/20140202_1600_kazrbaecc_peakTree.nc4 --no-nodes 3 --range-interval 400,3000

#python3 plot2d.py output/20170629_0830_Pol_peakTree.nc4 --range-interval 400,3243 --no-nodes 2

#python3 convert_to_json.py output/20170629_0830_Pol_peakTree.nc4 output/20170629_0830_data --time-interval 0-450 --range-interval 0-100
python3 convert_to_json.py output/20140221_2232_kazrbaeccpeako_peakTree.nc4 output/20140221_2232_kazrbaeccpeako_data --time-interval 0-410 --range-interval 0-100

#python3 convert_to_json.py output/20140202_1600_kazrbaeccpeako_peakTree.nc4 output/20140202_1600_kazrbaeccpeako_peakTree --time-interval 450-1235 --range-interval 0-79

