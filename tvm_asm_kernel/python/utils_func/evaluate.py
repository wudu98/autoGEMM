import tvm
from tvm import te
from tvm import testing
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity

import numpy as np
from template.asm_micro_kernel_template import matmul


def evaluate(M, K, N, record_file, target="llvm"):
    ctx = tvm.cpu(0)
    dtype = "float32"

    with autotvm.apply_history_best(record_file):
        with tvm.target.Target(target):
            s, arg_buf = matmul(M, K, N)
            func = tvm.build(s, arg_buf, target=tvm.target.Target(target))

            a = tvm.nd.array(np.random.uniform(-1, 1, size=(M, K)).astype(dtype), ctx)
            b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)

            workload = autotvm.task.args_to_workload(
                [M, K, N], "matmul"
            )
            tgt = tvm.target.Target.current()
            cfg = autotvm.task.DispatchContext.current.query(tgt, workload)

            bn = cfg["tile_y"].size[-1]
            kn = cfg["tile_k"].size[-1]
            B = te.placeholder((K, N), name="B")
            PackedB = te.compute(
                (K // kn, N // bn, kn, bn), lambda i, x, y, z: B[i * kn + y, x * bn + z], name="PackedB"
            )
            packed_b = tvm.nd.array(np.zeros((K // kn, N // bn, kn, bn), dtype=dtype), ctx)
            c = tvm.nd.array(np.zeros((M, N), dtype=dtype), ctx)
            bias = tvm.nd.array(np.random.rand(N,).astype(dtype), ctx)
            packb_schedule = te.create_schedule(PackedB.op)
            packb_func = tvm.build(packb_schedule, [B, PackedB], name="OP_GEMM_%dX%dX%d_packB" % (M, N, K), target=target)
            packb_func(b, packed_b)

            func(a, packed_b, c)
            # func(a, b, c)

    expected = np.dot(a.asnumpy(), b.asnumpy())

    tvm.testing.assert_allclose(c.asnumpy(), expected, rtol=1e-4, atol=1e-4)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=1000, min_repeat_ms=1000)
    mean_time = evaluator(a, packed_b, c).mean
    # mean_time = evaluator(a, b, c).mean
    gflops = 2 * M * N * K * 1e-9 / mean_time
    # print("%f" % (gflops))
    print("GFLOPS: %f, avg time: %f ms" % (gflops, mean_time * 1000))
