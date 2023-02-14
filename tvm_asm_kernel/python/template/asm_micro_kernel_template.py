import re
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task import ConfigEntity
from template.gen_asm_code.tvm_extern_asm_micro_kernel import intrin_gemm_MxKxN, gemm_MxKxN_impl

@autotvm.template("matmul")
def matmul(M, K, N, parallel, instruction):
    cfg = autotvm.get_config()

    # Tiling structure: split M/N/K into 3 axes each.
    cfg.define_split("tile_x", M, num_outputs=3)
    cfg.define_split("tile_y", N, num_outputs=3)
    cfg.define_split("tile_k", K, num_outputs=3)

    # Micro-kernel parameters used in tensorization.
    cfg.define_knob("nr_main_knob", [3, 4, 5])
    cfg.define_knob("MRSA_FLAG", [0, 1])
    if re.search(r"neon", instruction) :
        cfg.define_knob("unroll_k_knob", [8, 16, 32])
        cfg.define_knob("padding_size", [1, 4])
    elif re.search(r"sve", instruction) :
        cfg.define_knob("unroll_k_knob", [4, 8, 16])
        cfg.define_knob("padding_size", [1, 4, 16])

    padding_size = cfg["padding_size"].val

    # Matrix "A" has a shape of (M, K).
    A = te.placeholder((M, K), name="A")

    # Matrix "PackedB" has been pre-packed into shape of (K // kn, N // bn, K, bn_ceil), for "bn" is the innermost axis of the splited "N" dim.
    # "bn_ceil" is a padding size to store "bn" elements.
    # Note the pre-pack format is only available for inference mode, where weight matrix "B" is fixed.
    bn = cfg["tile_y"].size[-1]
    kn = cfg['tile_k'].size[-1]
    bn_ceil = ((bn - 1) // padding_size + 1) * padding_size

    PackedB = te.placeholder((K // kn, N // bn, kn, bn_ceil), name='PackedB')

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

    # Make (yi, xi, ki) the inner most axes, to be tensorized later.
    s[C].reorder(yt, kt, xt, yo, ko, xo, yi, xi, ki)

    # Let autotvm to find the best order of the 6 axes:
    cfg.define_reorder("reorder_outer", [yt, kt, xt, yo, ko, xo], "all")
    new_order = cfg["reorder_outer"].apply(s, C, [yt, kt, xt, yo, ko, xo])

    if parallel :
        # Fuse the outmost non-reducution axes.
        sibling_axes = []
        for axis in new_order:
            if axis not in [kt, ko]:
                sibling_axes.append(axis)
            else:
                break

        parallel_axis = s[C].fuse(*sibling_axes)

        assert parallel_axis is not None
        s[C].parallel(parallel_axis)

    pragma_axis = parallel_axis if parallel else xo

    # Inner kernel implementation for the tensorization.
    micro_kernel, uniq_id = intrin_gemm_MxKxN(
                                cfg["tile_x"].size[-1],
                                cfg["tile_k"].size[-1], 
                                cfg["tile_y"].size[-1],
                                K,
                                bn_ceil,
                                N,
                                )
    s[C].tensorize(yi, micro_kernel)
    s[C].pragma(pragma_axis, "import_llvm", gemm_MxKxN_impl(
                                cfg["tile_x"].size[-1],
                                cfg["tile_k"].size[-1], 
                                cfg["tile_y"].size[-1],
                                K,
                                bn_ceil,
                                N,
                                cfg["unroll_k_knob"].val,
                                cfg["nr_main_knob"].val,
                                cfg["MRSA_FLAG"].val,
                                instruction,
                                uniq_id
                                ))

    return s, [A, PackedB, C]
