#!/usr/bin/env bash

N=`seq 12 1 12`
M=`seq 100 100 1000`

for i in ${N[*]}; do
    for j in ${M[*]}; do
        echo 'Starting Iteration '$i' '$j' Now'
        python ./examples/28_2d_sep_vary_max_bond_dim.py $i $j > 'data_'$i'_'$j'.log' 
    done
done
