#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

UNROLL=8
TOT_REPEAT=65536000000

M_list=( 5 80 80)
N_list=(64 16 64)
K_list=(4 16 64 256)

for K in ${K_list[*]}
do
	LOOP_NUM=${#M_list[@]}
	for (( i=0; i<$LOOP_NUM; i++))
	do
		M=${M_list[$i]} 
		N=${N_list[$i]} 

		REPEAT=`expr $TOT_REPEAT / $M / $N / $K`
		if test $REPEAT -gt 1000000000
		then
			REPEAT=1000000000
		fi
		NR=4
		
		echo -n $M, $N, $K," " 

		python make_c_file_instrinsic.py $M $N $K $UNROLL $NR $REPEAT
		make -s
		./benchmark_kernel

		python make_c_file_asm_pipeline_expreiment.py $M $N $K $UNROLL $NR $REPEAT 0
		make -s
		./benchmark_kernel

		python make_c_file_asm_pipeline_expreiment.py $M $N $K $UNROLL $NR $REPEAT 1
		make -s
		./benchmark_kernel

		python make_c_file_asm_pipeline_expreiment.py $M $N $K $UNROLL $NR $REPEAT 2
		make -s
		./benchmark_kernel

		python make_c_file_asm_pipeline_expreiment.py $M $N $K $UNROLL $NR $REPEAT 3
		make -s
		./benchmark_kernel

		echo ""
	done
done
