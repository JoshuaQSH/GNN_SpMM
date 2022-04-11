#!/bin/bash

ROW=$(seq 1000 200 20000)
# The ROW and COL can be the same at once
COL=$(seq 1000 200 20000)
DEVID=${1-0}
DENSITY=$(seq 0 0.1 0.9)
FORMAT=(csr csc coo)

startime=$(date +%s)
for size in ${ROW}
do
    for den in ${DENSITY}
    do
	for form in ${FORMAT[@]}
	do
            echo ">> Running in GPU ${DEVID} and running the size with Row: ${size}, Density: ${den} ..."
	    echo "> Storing Format: ${form} ..."
            python spmm_test.py --gpuid ${DEVID} \
		            --row-dim ${size} \
		            --col-dim ${size} \
		            --density ${den} \
			    --format ${form} \
			    #2>&1 | tee cupy_log_spmm.log
	done
    done
done

echo "Testing SpMM done!"
endtime=$(date +%s)

echo "Runtime: $((endtime - startime))"
