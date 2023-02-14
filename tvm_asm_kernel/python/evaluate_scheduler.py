import tvm
from tvm import te
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
    parser.add_argument("-a", "--arch", default="mac", choices=["mac", "linux", "a64fx"], help='select architecture mac or linux')
    parser.add_argument("--parallel", action="store_true", help='whether parallel execute')
    parser.add_argument("--scheduler_log", type=str, required=True)
    args = parser.parse_args()

    M = args.m
    K = args.k
    N = args.n
    parallel = args.parallel
    best_schedule_file = args.scheduler_log

    if args.arch == "mac" :
        instruction = "neon"
        target = f"llvm -mtriple=arm64-apple-darwin -mattr=+{instruction}"
    elif args.arch == "linux" :
        instruction = "neon"
        target = f"llvm -mtriple=aarch64-linux-gnu -mattr=+{instruction}"
    elif args.arch == "a64fx" :
        instruction = "sve"
        target = f"llvm -mtriple=aarch64-linux-gnu -mattr=+{instruction}"
        
    # print('%d, %d, %d' % (M, N, K))
    evaluate(M, K, N, best_schedule_file, parallel, pack_dso=True, instruction=instruction, target=target)
