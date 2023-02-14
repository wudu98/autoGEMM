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
from utils_func.tune import tune


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run autotvm.")
    parser.add_argument("-m", type=int, required=True, help="M")
    parser.add_argument("-k", type=int, required=True, help="K")
    parser.add_argument("-n", type=int, required=True, help="N")
    parser.add_argument("-a", "--arch", default="mac", choices=["mac", "linux", "a64fx"], help='select architecture mac or linux')
    parser.add_argument("--parallel", action="store_true", help='whether parallel execute')
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        required=False,
        default=2000,
        help="Step of autotvm search.",
    )
    parser.add_argument(
        "-r",
        "--record_file",
        default="matmul.log",
        type=str,
        required=False,
        help="Specify name of the file to record autotvm tuning result",
    )
    args = parser.parse_args()

    M = args.m
    K = args.k
    N = args.n

    record_file = args.record_file
    step = args.step
    parallel = args.parallel

    if args.arch == "mac" :
        instruction = "neon"
        target = f"llvm -mtriple=arm64-apple-darwin -mattr=+{instruction}"
    elif args.arch == "linux" :
        instruction = "neon"
        target = f"llvm -mtriple=aarch64-linux-gnu -mattr=+{instruction}"
    elif args.arch == "a64fx" :
        instruction = "sve"
        target = f"llvm -mtriple=aarch64-linux-gnu -mattr=+{instruction}"
        
    tune(M, K, N, record_file, parallel, n_trial=step, instruction=instruction, target=target)
    evaluate(M, K, N, record_file, parallel, pack_dso=False, instruction=instruction, target=target)
