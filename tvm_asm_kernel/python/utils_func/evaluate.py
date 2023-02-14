import os
import json
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity

import numpy as np
from template.asm_micro_kernel_template import matmul

def evaluate(M, K, N, record_file, parallel, pack_dso, instruction="neon", target="llvm"):
    ctx = tvm.cpu(0)
    dtype = "float32"

    with autotvm.apply_history_best(record_file):
        with tvm.target.Target(target):
            s, arg_buf = matmul(M, K, N, parallel, instruction)
            func = tvm.build(s, arg_buf, name="OP_GEMM_%dX%dX%d" % (M, N, K), target=tvm.target.Target(target))
            # print(tvm.lower(s, arg_buf))

            a = tvm.nd.array(np.random.uniform(-1, 1, size=(M, K)).astype(dtype), ctx)
            b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)
            c = tvm.nd.array(np.zeros((M, N), dtype=dtype), ctx)

            workload = autotvm.task.args_to_workload(
                [M, K, N, parallel, instruction], "matmul"
            )
            tgt = tvm.target.Target.current()
            cfg = autotvm.task.DispatchContext.current.query(tgt, workload)

            padding_size = cfg["padding_size"].val
            bn = cfg["tile_y"].size[-1]
            kn = cfg["tile_k"].size[-1]
            bn_ceil = ((bn - 1) // padding_size + 1) * padding_size

            B = te.placeholder((K, N), name="B")
            PackedB = te.compute(
                (K // kn, N // bn, kn, bn_ceil), 
                lambda i, x, y, z: te.if_then_else(
                    z < bn, B[i * kn + y, x * bn + z], 0
                ), name="PackedB"
            )

            packed_b = tvm.nd.array(np.zeros((K // kn, N // bn, kn, bn_ceil), dtype=dtype), ctx)
            packb_schedule = te.create_schedule(PackedB.op)
            bigK, bigN, _, littleN = packb_schedule[PackedB].op.axis
            packb_schedule[PackedB].vectorize(littleN)
            if parallel:
                parallel_axis = packb_schedule[PackedB].fuse(bigK, bigN)
                packb_schedule[PackedB].parallel(parallel_axis)
            packb_func = tvm.build(packb_schedule, [B, PackedB], name="OP_GEMM_%dX%dX%d_packB" % (M, N, K), target=target)
            # print(tvm.lower(packb_schedule, [B, PackedB]))

            packb_func(b, packed_b)
            func(a, packed_b, c)

    expected = np.dot(a.asnumpy(), b.asnumpy())

    np.testing.assert_allclose(c.asnumpy(), expected, rtol=1e-2, atol=1e-4)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=1000, min_repeat_ms=5000)
    mean_time = evaluator(a, packed_b, c).mean
    gflops = 2 * M * N * K * 1e-9 / mean_time

    print("TVM offline GFLOPS: %f, avg time: %f ms" % (gflops, mean_time * 1000))

    if pack_dso:
        packb_func.save(f"../build/gemm_obj/{packb_func.name}.o")
        func.save(f"../build/gemm_obj/{func.name}.o")
        os.system(f"ar rcs ../build/library/GEMM_{M}X{N}X{K}_kernel.a ../build/gemm_obj/OP_GEMM_{M}X{N}X{K}*.o")
        func.import_module(packb_func)
        func.export_library(f"../build/library/GEMM_{M}X{N}X{K}_kernel.so")
