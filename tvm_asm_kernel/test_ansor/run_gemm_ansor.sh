#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

tune_output_path="tune_output/log"

arch=$1
threads=$2
tune=$3

export OMP_NUM_THREADS=${threads}

if [ "${tune}" == "1" ]; then
    use_tune="--use_tune"
    if [[ -d $tune_output_path ]]; then
        rm -rf $tune_output_path
    fi
else
    use_tune=""
fi

MNK_file=${PROJECT_ROOT}/../MNK.txt 
cnt=0
cat $MNK_file | while read line
do
    M=`echo $line | awk '{print $1}'`
    N=`echo $line | awk '{print $2}'`
    K=`echo $line | awk '{print $3}'`
    python gemm_ansor.py -m ${M} -n ${N} -k ${K} -a ${arch} -r $tune_output_path/${cnt}_matmul_${M}_${N}_${K}.json ${use_tune}
    let cnt+=1
done

