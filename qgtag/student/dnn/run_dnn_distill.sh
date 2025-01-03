#!/bin/bash                                                                                                                                                                                                 

N=10

for ((i=0;i<N;i++))
do
    echo "$(((i+1))) of $N"
    python3 dnn_distill.py \
	    -nEpochs=200 \
	    -batchSize=500 \
	    -latentSize=128 \
	    -phiSizes=250 \
	    -doEarlyStopping \
	    -patience=10 \
	    -usePIDs \
	    -nLayers=2 \
	    -layerSize=100 \
	    -ModelNum=$((i)) \
	    -alpha=0.01
done
