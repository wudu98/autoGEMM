#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/..; pwd`
cd ${PROJECT_ROOT}

UNROLL=8
TOT_REPEAT=65536000000

for M in $(seq 1 1 128)
do
	N=$M
	K=$M
	REPEAT=`expr $TOT_REPEAT / $M / $N / $K`
	if test $REPEAT -gt 1000000000
	then
		REPEAT=1000000000
	fi
	for NR in $(seq 3 1 5)
	do
		python python/make_c_file_asm.py $M $N $K $UNROLL $NR $REPEAT
		# python python/make_c_file_asm_sve.py $M $N $K $UNROLL $NR $REPEAT
		make -s
		./benchmark_kernel
	done
done
