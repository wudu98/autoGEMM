import tvm
from tvm import te
from tvm import testing
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity
from template.gen_asm_code.tvm_extern_asm_micro_kernel import intrin_gemm_MxKxN, gemm_MxKxN_impl

dtype = "float32"

@autotvm.template("matmul")
def matmul(M, K, N):
    cfg = autotvm.get_config()

    # Tiling structure: split M/N/K into 3 axes each.
    cfg.define_split("tile_x", M, num_outputs=3)
    cfg.define_split("tile_y", N, num_outputs=3)
    cfg.define_split("tile_k", K, num_outputs=3)

    cfg.define_knob("unroll_k_knob", [8, 16, 32])
    cfg.define_knob("nr_main_knob", [3, 4, 5])

    # Matrix "A" has a shape of (M, K).
    A = te.placeholder((M, K), name="A")

    # Matrix "PackedB" has been pre-packed into shape of (N // bn, K, bn), for "bn" is the innermost axis of the splited "N" dim.
    # Note the pre-pack format is only available for inference mode, where weight matrix "B" is fixed.
    bn = cfg["tile_y"].size[-1]
    kn = cfg['tile_k'].size[-1]
    PackedB = te.placeholder((K // kn, N // bn, kn, bn), name='PackedB')
    # B = te.placeholder((K, N), name="B")
    # PackedB = te.compute((K // kn, N // bn, K, bn), lambda w, x, y, z: B[w * kn + y, x * bn + z], name='PackedB')

    k = te.reduce_axis((0, K), "k")

    # C = A x PackedB:
    C = te.compute(
        (M, N),
        lambda x, y: te.sum(A[x, k] * PackedB[k // kn, y // bn, k % kn, y % bn], axis=k),
        name="C",
    )

    # Schedule:
    s = te.create_schedule(C.op)
    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis

    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    kt, ko, ki = cfg["tile_k"].apply(s, C, k)

    # Make (xi, yi, ki) the inner most axes, to be tensorized later.
    s[C].reorder(kt, xo, yt, xt, yo, ko, xi, yi, ki)

    # Let autotvm to find the best order of the outmost 6 axes:
    cfg.define_reorder("reorder", [kt, xo, yt, xt, yo, ko], "all")
    new_order = cfg["reorder"].apply(s, C, [kt, xo, yt, xt, yo, ko])

    # Inner kernel implementation for the tensorization.
    micro_kernel, uniq_id = intrin_gemm_MxKxN(
                                cfg["tile_x"].size[-1],
                                cfg["tile_k"].size[-1], 
                                cfg["tile_y"].size[-1],
                                K,
                                cfg["tile_y"].size[-1],
                                N,
                                )
    s[C].tensorize(xi, micro_kernel)
    s[C].pragma(yo, "import_llvm", gemm_MxKxN_impl(
                                cfg["tile_x"].size[-1],
                                cfg["tile_k"].size[-1], 
                                cfg["tile_y"].size[-1],
                                K,
                                cfg["tile_y"].size[-1],
                                N,
                                cfg["unroll_k_knob"].val,
                                cfg["nr_main_knob"].val,
                                uniq_id
                                ))
    return s, [A, PackedB, C]
    # return s, [A, B, C]
