#!/bin/bash                                                                                                                                                                                                 

N=1

for ((i=0;i<N;i++))
do
    echo "$(((i+1))) of $N"
    python3 pfn_mix_train.py \
	    -nEpochs=200 \
	    -batchSize=500 \
	    -latentSize=128 \
	    -phiSizes=250 \
	    -doEarlyStopping \
	    -patience=10 \
	    -usePIDs 
done
