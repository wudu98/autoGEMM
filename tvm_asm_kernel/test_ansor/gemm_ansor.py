import os
import sys
import argparse
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi, testing

parser = argparse.ArgumentParser(description="Script to run ansor.")
parser.add_argument("-m", type=int, required=True, help="M")
parser.add_argument("-k", type=int, required=True, help="K")
parser.add_argument("-n", type=int, required=True, help="N")
parser.add_argument("-a", "--arch", default="mac", choices=["mac", "linux", "a64fx"], help='select architecture mac or linux')
parser.add_argument("--use_tune", action="store_true", help='whether parallel execute')
parser.add_argument(
    "-r",
    "--record_file",
    default="matmul.log",
    type=str,
    required=False,
    help="Specify name of the file to record ansor tuning result",
)
args = parser.parse_args()

M = args.m
N = args.n
K = args.k
dtype = "float32"

if args.arch == "mac" :
    instruction = "neon"
    target = f"llvm -mtriple=arm64-apple-darwin -mattr=+{instruction}"
elif args.arch == "linux" :
    instruction = "neon"
    target = f"llvm -mtriple=aarch64-linux-gnu -mattr=+{instruction}"
elif args.arch == "a64fx" :
    instruction = "sve"
    target = f"llvm -mtriple=aarch64-linux-gnu -mattr=+{instruction}"

@auto_scheduler.register_workload
def gemm_ansor(M, N, K):
    k_axis = te.reduce_axis((0, K), name='k')
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y:te.sum(A[x, k_axis] * B[k_axis, y], axis=k_axis), name='C')
    return [A, B, C]

task = auto_scheduler.SearchTask(
    func=gemm_ansor, args=(M, N, K), target=target
)

log_file = args.record_file

tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10000, 
    runner=auto_scheduler.LocalRunner(number=100, repeat=1, timeout=300, min_repeat_ms=100),#, enable_cpu_cache_flush=True),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
)

if args.use_tune == 1:
    task.tune(tune_option)

sch, args = task.apply_best(log_file)

func = tvm.build(sch, args, target)
#print(tvm.lower(sch, args))

# Check correctness
ctx = tvm.device(target)
a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)
c = tvm.nd.array(np.zeros((M, N), dtype=dtype), ctx)
answer = np.dot(a.asnumpy(), b.asnumpy())
func(a, b, c)

# Check results
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-2, atol=1e-4)
# Evaluate execution time
evaluator = func.time_evaluator(func.entry_name, ctx, number=1000, min_repeat_ms=5000)
latency = evaluator(a, b, c).mean
print('M=%d, N=%d, K=%d, time: %f ms, GFLOPS: %f' % (M, N, K, latency * 1000, 2 * M * N * K / latency / 1e9))
