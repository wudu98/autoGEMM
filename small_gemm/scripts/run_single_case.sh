#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/..; pwd`
cd ${PROJECT_ROOT}

M=$1
N=$2
K=$3
UNROLL=$4
NR=$5
TOT_REPEAT=65536000000

REPEAT=`expr $TOT_REPEAT / $M / $N / $K`
if test $REPEAT -gt 1000000000
then
	REPEAT=1000000000
fi

python python/make_c_file_asm.py $M $N $K $UNROLL $NR $REPEAT
# python python/make_c_file_asm_sve.py $M $N $K $UNROLL $NR $REPEAT
make -s
./benchmark_kernel
