#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

UNROLL=8
TOT_REPEAT=65536000000

M_list=(20 80)
N_list=(16 32 64)
K_list=(4 8 16 32 64 128)

for K in ${K_list[*]}
do
	for M in ${M_list[*]}
	do
		for N in ${N_list[*]}
		do
			REPEAT=`expr $TOT_REPEAT / $M / $N / $K`
			if test $REPEAT -gt 1000000000
			then
				REPEAT=1000000000
			fi
			NR=4
			
			python make_c_file_base.py $M $N $K $UNROLL $NR $REPEAT
			make -s
			./benchmark_kernel

			python make_c_file_asm_v1.py $M $N $K $UNROLL $NR $REPEAT
			make -s
			./benchmark_kernel

			python make_c_file_asm_v2.py $M $N $K $UNROLL $NR $REPEAT
			make -s
			./benchmark_kernel

			python make_c_file_asm_v3.py $M $N $K $UNROLL $NR $REPEAT
			make -s
			./benchmark_kernel

			python make_c_file_asm_v4_all.py $M $N $K $UNROLL $NR $REPEAT
			make -s
			./benchmark_kernel

		done
	done
done
