#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/..; pwd`
cd ${PROJECT_ROOT}

tune_num=$1

M=(64    64   64   256  64   128 128  512 512 128 512 256 256  1024 1024 256  512  512  2048 2048 512)
N=(12544 3136 3136 3136 3136 784 784  784 784 784 784 196 196  196  196  196  49   49   49   49   49)
K=(147   64   576  64   256  256 1152 128 256 512 128 512 2304 256  512  1024 1024 4608 512  1024 2048)

MNK_file=${PROJECT_ROOT}/MNK.txt 
if [[ -f $MNK_file ]]; then
    rm -rf $MNK_file
fi

for (( i=0; i<21; i++))
do
	echo ${M[$i]} ${N[$i]} ${K[$i]} >> MNK.txt
done

bash ./scripts/tune.sh $tune_num

if [[ -f "tune_output/tune.over" ]]; then
	if [[ -f "resnet50_scheduler" ]]; then
		rm -rf resnet50_scheduler
	fi
	mkdir resnet50_scheduler
    cp tune_output/scheduler_summary.log resnet50_scheduler/scheduler_summary.log
fi

