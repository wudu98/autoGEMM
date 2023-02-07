#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/..; pwd`
cd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$PWD/python
export TVM_CC=clang++

build_output_path="build"
tune_output_path="tune_output"
parallel="--parallel"
offline="--offline"

if [[ -d $build_output_path ]]; then
    rm -rf $build_output_path
fi
mkdir -p $build_output_path

MNK_file=${PROJECT_ROOT}/MNK.txt 
scheduler_log=${tune_output_path}/scheduler_summary.log
scheduler_log_output=$build_output_path/scheduler_summary.log

python ${PROJECT_ROOT}/python/summarize_scheduler.py --input $scheduler_log --output $scheduler_log_output

cd $build_output_path

python ${PROJECT_ROOT}/python/evaluate_scheduler.py \
    --MNK_file $MNK_file \
    --scheduler_log scheduler_summary.log \
    ${tune_num} \
    ${parallel}

touch build.over
