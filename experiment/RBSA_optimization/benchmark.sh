#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

UNROLL=8
TOT_REPEAT=65536000000

M_list=(20 21 22 23 24 25 25 25 25 25 21 26 31 36)
N_list=(32 32 32 32 32 32 36 40 44 48 20 36 52 68)

LOOP_NUM=${#M_list[@]}
for (( i=0; i<$LOOP_NUM; i++))
do
	M=${M_list[$i]} 
	N=${N_list[$i]} 
	K=64

	REPEAT=`expr $TOT_REPEAT / $M / $N / $K`
	if test $REPEAT -gt 1000000000
	then
		REPEAT=1000000000
	fi
	NR=4
	
	python make_c_file_asm_v1.py $M $N $K $UNROLL $NR $REPEAT
	make -s
	./benchmark_kernel

	python make_c_file_asm_v2.py $M $N $K $UNROLL $NR $REPEAT
	make -s
	./benchmark_kernel

	python make_c_file_asm_v3.py $M $N $K $UNROLL $NR $REPEAT
	make -s
	./benchmark_kernel

done
