#!/usr/bin/env bash

N=`seq 12 1 12`
M=`seq 400 100 600`

for i in ${N[*]}; do
    for j in ${M[*]}; do
        echo 'Starting Iteration '$i' '$j' Now'
        python ./examples/22_2d_sep_vary_max_bond_dim.py $i $j 2>&1 | tee 'data_'$i'_'$j'.log' 
    done
done
