#!/bin/bash
set -e

tmp=`dirname $0`
PROJECT_ROOT=`cd $tmp; pwd`
cd ${PROJECT_ROOT}

export TVM_NUM_THREADS=1
USE_TUNE=$1
filename=../MNK.txt
cat $filename | while read line
do
    python gemm_ansor.py ${line} ${USE_TUNE}
done

