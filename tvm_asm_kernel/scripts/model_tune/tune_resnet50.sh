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

M=(64    64   64   256  64   128 128  512 512 128 256 256  1024 1024 256  512  512  2048 2048 512)
N=(12544 3136 3136 3136 3136 784 784  784 784 784 196 196  196  196  196  49   49   49   49   49)
K=(147   64   576  64   256  256 1152 128 256 512 512 2304 256  512  1024 1024 4608 512  1024 2048)

MNK_file=${PROJECT_ROOT}/MNK.txt 
if [[ -f $MNK_file ]]; then
    rm -rf $MNK_file
fi

for (( i=0; i<20; i++))
do
	echo ${M[$i]} ${N[$i]} ${K[$i]} >> MNK.txt
done

bash ./scripts/utils/tune.sh $arch $threads $tune_num 

if [[ -f "tune_output/tune.over" ]]; then
	if [[ -f "scheduler_house/resnet50" ]]; then
		rm -rf scheduler_house/resnet50
	fi
	mkdir -p scheduler_house/resnet50
    cp tune_output/scheduler_summary.log scheduler_house/resnet50/scheduler_summary.log
fi

