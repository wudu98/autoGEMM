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
    parser.add_argument("-m", type=int, required=True, help="M")
    parser.add_argument("-k", type=int, required=True, help="K")
    parser.add_argument("-n", type=int, required=True, help="N")
    parser.add_argument("--parallel", action="store_true", help='whether parallel execute')
    parser.add_argument("--scheduler_log", type=str, required=True)
    args = parser.parse_args()

    M = args.m
    K = args.k
    N = args.n
    parallel = args.parallel
    best_schedule_file = args.scheduler_log

    from config.mac_config import target
    # print('%d, %d, %d' % (M, N, K))
    evaluate(M, K, N, best_schedule_file, parallel, pack_dso=False,target=target)
