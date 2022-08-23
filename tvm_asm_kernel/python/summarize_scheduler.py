import tvm
from tvm import autotvm

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    autotvm.record.pick_best(input_file, output_file)