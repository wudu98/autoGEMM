#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/..; pwd`
cd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$PWD/python
export TVM_CC=clang++

tune_output_path="tune_output"
tune_num=$1

if [[ -d $tune_output_path ]]; then
    rm -rf $tune_output_path
fi
mkdir -p $tune_output_path
mkdir -p $tune_output_path/perf
mkdir -p $tune_output_path/log

touch $tune_output_path/scheduler_summary.log

MNK_file=${PROJECT_ROOT}/MNK.txt 
cnt=0
cat $MNK_file | while read line
do
    M=`echo $line | awk '{print $1}'`
    N=`echo $line | awk '{print $2}'`
    K=`echo $line | awk '{print $3}'`
    python ${PROJECT_ROOT}/python/tune_scheduler.py -m ${M} -n ${N} -k ${K} -s ${tune_num} -r $tune_output_path/matmul.log > $tune_output_path/perf/${cnt}_matmul_${M}_${N}_${K}.perf
    mv $tune_output_path/matmul.log.tmp $tune_output_path/log/${cnt}_matmul_${M}_${N}_${K}.log
    python ${PROJECT_ROOT}/python/summarize_scheduler.py --input $tune_output_path/matmul.log --output $tune_output_path/scheduler_summary.log
    let cnt+=1
done

touch $tune_output_path/tune.over
