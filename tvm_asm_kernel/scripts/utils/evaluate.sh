#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp/../..; pwd`
cd ${PROJECT_ROOT}

export PYTHONPATH=$PYTHONPATH:$PWD/python
export TVM_CC=clang++

build_output_path="build"
tune_output_path="tune_output"

arch=$1
threads=$2
repeats=$3

if [ "${threads}" == "1" ]; then
    parallel=""
elif [ "${threads}" -gt "1" ]; then
    parallel="--parallel"
else
    echo "threads num error"
    exit -1
fi

export OMP_NUM_THREADS=${threads}

if [[ -d $build_output_path ]]; then
    rm -rf $build_output_path
fi
mkdir -p $build_output_path
mkdir -p $build_output_path/gemm_obj
mkdir -p $build_output_path/library

MNK_file=${PROJECT_ROOT}/MNK.txt 
scheduler_log=${tune_output_path}/scheduler_summary.log
scheduler_log_output=$build_output_path/scheduler_summary.log

python ${PROJECT_ROOT}/python/summarize_scheduler.py --input $scheduler_log --output $scheduler_log_output
python ${PROJECT_ROOT}/python/build_kernel_params_list.py clean
python ${PROJECT_ROOT}/python/build_kernel_params_list.py

cd src
make -s

MNK_file=${PROJECT_ROOT}/MNK.txt 
cnt=0
cat $MNK_file | while read line
do
    M=`echo $line | awk '{print $1}'`
    N=`echo $line | awk '{print $2}'`
    K=`echo $line | awk '{print $3}'`
    python ${PROJECT_ROOT}/python/evaluate_scheduler.py -m ${M} -n ${N} -k ${K} -a ${arch} ${parallel} --scheduler_log ${PROJECT_ROOT}/${scheduler_log_output}
    ./benchmark_kernel ${M} ${N} ${K} ${repeats}
    let cnt+=1
done

cd ../${build_output_path}
touch build.over
