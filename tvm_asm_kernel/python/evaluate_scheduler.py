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
    parser.add_argument("--parallel", action="store_true", help='whether parallel execute')
    parser.add_argument("--offline", action="store_true", help='whether to use offline PackB')
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
    
    offline_pack = args.offline
    parallel = args.parallel

    from config.mac_config import target
    for i in range(len(M)):
        # print('%d, %d, %d' % (M[i], N[i], K[i]))
        evaluate(M[i], K[i], N[i], best_schedule_file, offline_pack, parallel, target=target)
