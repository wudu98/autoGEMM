#!/bin/bash

UNROLL=8
NR=4
TOT_REPEAT=65536000000

for M in $(seq 1 1 120)
do
	N=$M
	K=$M
	REPEAT=`expr $TOT_REPEAT / $M / $N / $K`
	if test $REPEAT -gt 1000000000
	then
		REPEAT=1000000000
	fi
	python make_c_file_asm.py $M $N $K $UNROLL $NR $REPEAT
	make -s
	./benchmark_kernel
done
