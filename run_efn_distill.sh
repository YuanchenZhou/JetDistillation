#!/bin/bash                                                                                                                                                                                                 

N=1

for ((i=0;i<N;i++))
do
    echo "$(((i+1))) of $N"
    python3 efn_distill.py \
	    -nEpochs=200 \
	    -batchSize=500 \
	    -latentSizeTeacher=128 \
	    -phiSizesTeacher=250 \
	    -latentSizeStudent=128 \
	    -phiSizesStudent=100 \
	    -doEarlyStopping \
	    -patience=10 \
	    -usePIDs 
done
