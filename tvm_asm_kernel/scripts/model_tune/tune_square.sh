#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../..; pwd`
cd ${PROJECT_ROOT}

arch=$1
threads=$2
tune_num=$3

if [ "${threads}" == "1" ]; then
    parallel=""
elif [ "${threads}" -gt "1" ]; then
    parallel="--parallel"
else
    echo "threads num error"
    exit -1
fi

export OMP_NUM_THREADS=${threads}

M=(64 128 256 512 1024 2048)
N=(64 128 256 512 1024 2048)
K=(64 128 256 512 1024 2048)

MNK_file=${PROJECT_ROOT}/MNK.txt 
if [[ -f $MNK_file ]]; then
    rm -rf $MNK_file
fi

for (( i=0; i<6; i++))
do
	echo ${M[$i]} ${N[$i]} ${K[$i]} >> MNK.txt
done

bash ./scripts/utils/tune.sh $arch $threads $tune_num 

if [[ -f "tune_output/tune.over" ]]; then
	if [[ -f "scheduler_house/square" ]]; then
		rm -rf scheduler_house/square
	fi
	mkdir -p scheduler_house/square
    cp tune_output/scheduler_summary.log scheduler_house/square/scheduler_summary.log
fi

