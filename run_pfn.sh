#!/bin/bash

N=2

for ((i=0;i<N;i++))
do
    python3 pfn_buffer.py  -num_model=$N -num_epoch=100 -model_number=$i -n_power=7 -batch_size=500
done
