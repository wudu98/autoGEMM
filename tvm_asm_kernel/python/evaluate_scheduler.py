import tvm
from tvm import te
from tvm import testing
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity

import os
import sys
import random
import string
import argparse
import numpy as np

from template.asm_micro_kernel_template import matmul
from utils_func.evaluate import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--MNK_file", type=str, required=True)
    parser.add_argument("--scheduler_log", type=str, required=True)
    args = parser.parse_args()

    M=[]
    N=[]
    K=[]
    best_schedule_file = args.scheduler_log
    with open(args.MNK_file, "r") as f:
        for line in f:
            MNK = line.strip().split(" ")
            MNK = [int(x) for x in MNK]
            M.append(MNK[0])
            N.append(MNK[1])
            K.append(MNK[2])

    mod_list = []
    
    # target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
    target = "llvm -mtriple=arm64-apple-darwin -mattr=+neon"
    # target = "llvm"
    for i in range(len(M)):
        # print('%d, %d, %d' % (M[i], N[i], K[i]))
        evaluate(M[i], K[i], N[i], best_schedule_file, target=target)
