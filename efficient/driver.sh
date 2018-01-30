#!/usr/bin/env bash

N=`seq 6 2 14`
M=`seq 100 100 1000`

for i in ${N[*]}; do
    for j in ${M[*]}; do
        python ./examples/22_2d_sep_vary_max_bond_dim.py $i $j > 'data_'$i'_'$j'.log' 
    done
done


for i in ${BL[*]}; do
    # Run doped 10e
    python ./code/hf/src_metal_k.py $NAT $i 5 > './data/logs/vdz/rhf/nat_'$NAT'_bl_'$i'_doped_5.log' &
    #mv chkf './data/chkfiles/vdz/rhf/nat_'$NAT'_bl_'$i'_doped_5.chk'
    echo 'Completed calculation at BL='$i' Doped 5'
    # Run doped 6e
    python ./code/hf/src_metal_k.py $NAT $i 3 > './data/logs/vdz/rhf/nat_'$NAT'_bl_'$i'_doped_3.log' &
    #mv chkf './data/chkfiles/vdz/rhf/nat_'$NAT'_bl_'$i'_doped_3.chk'
    echo 'Completed calculation at BL='$i' Doped 3'
    # Run doped 2e
    python ./code/hf/src_metal_k.py $NAT $i 1 > './data/logs/vdz/rhf/nat_'$NAT'_bl_'$i'_doped_1.log' &
    #mv chkf './data/chkfiles/vdz/rhf/nat_'$NAT'_bl_'$i'_doped_1.chk'
    echo 'Completed calculation at BL='$i' Doped 1'
    # Run doped 0e
    NAT=$(($NAT+2))
    python ./code/hf/src_metal_k.py $NAT $i 0 > './data/logs/vdz/rhf/nat_'$NAT'_bl_'$i'_doped_0.log'
    #mv chkf './data/chkfiles/vdz/rhf/nat_'$NAT'_bl_'$i'_doped_0.chk'
    echo 'Completed calculation at BL='$i' Doped 0'
    NAT=$(($NAT-2))
done
