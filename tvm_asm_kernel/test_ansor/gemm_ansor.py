import os
import sys
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
use_tune = int(sys.argv[4])
print('M=%d, N=%d, K=%d' % (M, N, K))

dtype = "float32"
# target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
target = "llvm -mtriple=arm64-apple-darwin -mattr=+neon"

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

log_file = f"./logs/gemm_{M}X{N}X{K}.json"

tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10000, 
    runner=auto_scheduler.LocalRunner(number=5, repeat=3, timeout=50, min_repeat_ms=200),#, enable_cpu_cache_flush=True),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
)

if use_tune == 1:
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
np.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-3)
# Evaluate execution time
evaluator = func.time_evaluator(func.entry_name, ctx, number=1000)
latency = evaluator(a, b, c).mean
print('time: %f ms, GFLOPS: %f' % (latency * 1000, 2 * M * N * K / latency / 1e9))
