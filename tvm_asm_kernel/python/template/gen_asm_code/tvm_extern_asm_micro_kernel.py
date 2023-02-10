import os
import re
import tvm
from tvm import te
from tvm.contrib import utils, clang
import random
import string

from config.common_config import cc_compiler

class GemmTensorIntrin(object):
    def __init__(self, M, K, N, lda, ldb, ldc, uniq_id, ins=None, outs=None):
        self.M = M
        self.N = N
        self.K = K

        self.lda = lda
        self.ldb = ldb
        self.ldc = ldc

        self.uniq_id = uniq_id

        self.ins = ins
        self.outs = outs

    def _body(self):
        ib = tvm.tir.ir_builder.create()
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                f"gemm_{self.M}x{self.K}x{self.N}_{self.lda}_{self.ldb}_{self.ldc}_xsmm_{self.uniq_id}",
                self.ins[0].access_ptr("r"),
                self.ins[1].access_ptr("r"),
                self.outs[0].access_ptr("w"),
                self.lda,
                self.ldb,
                self.ldc,
            )
        )
        return ib.get()

    def _init(self):
        return None

    def _update(self):
        ib = tvm.tir.ir_builder.create()
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                f"gemm_{self.M}x{self.K}x{self.N}_{self.lda}_{self.ldb}_{self.ldc}_xsmm_with_bias_{self.uniq_id}",
                self.ins[0].access_ptr("r"),
                self.ins[1].access_ptr("r"),
                self.outs[0].access_ptr("w"),
                self.lda,
                self.ldb,
                self.ldc,
            )
        )
        return ib.get()

    def body(self):
        return self._body(), self._init(), self._update()



def intrin_gemm_MxKxN(M, K, N, lda, ldb, ldc):
    """Defines a SIMD-accelerated transposed matmul."""
    # we generate a unique ID for every intrinsic definition, to prevent name
    # collisions in the generated source (e.g., if there are multiple operators
    # in the same module that use the same intrinsic)
    #
    # TODO(weberlo, areusch): to cut down on memory usage, we should cache each intrinsic
    # instantiation and include it only once, eliminating the need for unique
    # IDs
    UNIQ_ID_LEN = 8
    uniq_id = "".join(random.choices(string.ascii_uppercase, k=UNIQ_ID_LEN))

    a = te.placeholder((M, K), name='a')
    b = te.placeholder((K, N), name='b')
    k_axis = te.reduce_axis((0, K), name='k')
    
    c = te.compute(
        (M, N),
        lambda i, j: te.sum(a[i, k_axis] * b[k_axis, j], axis=k_axis),
        name="c",
    )
    a_buffer = tvm.tir.decl_buffer(a.shape, a.dtype, name='a_buffer', offset_factor=1, strides=[te.var('s1'), 1])
    b_buffer = tvm.tir.decl_buffer(b.shape, b.dtype, name='b_buffer', offset_factor=1, strides=[te.var('s2'), 1])
    c_buffer = tvm.tir.decl_buffer(c.shape, c.dtype, name='c_buffer', offset_factor=1, strides=[te.var('s3'), 1])
    bind_map = {
        a: a_buffer,
        b: b_buffer,
        c: c_buffer,
    }

    def intrin_func(ins, outs):
        intrin = GemmTensorIntrin(M, K, N, lda, ldb, ldc, uniq_id, ins, outs)
        return intrin.body()

    intrin_decl = te.decl_tensor_intrin(c.op, intrin_func, binds=bind_map)
    return intrin_decl, uniq_id


def gemm_MxKxN_impl(M, K, N, lda, ldb, ldc, unroll_k, nr_main, MRSA_FLAG, instruction, uniq_id):
    if re.search(r"neon", instruction) :
        from template.gen_asm_code.gen_xsmm_asm_armv8_neon_code import xsmm_asm_armv8_code
    elif re.search(r"sve", instruction) :
        from template.gen_asm_code.gen_xsmm_asm_armv8_sve_code import xsmm_asm_armv8_code

    # Create c source code
    cc_code = xsmm_asm_armv8_code(M, K, N, lda, ldb, ldc, unroll_k, nr_main, MRSA_FLAG, uniq_id)
    
    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # ll_path = "temp.ll"
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(cc_code, output=ll_path, options=["-march=armv8-a", "-O3", "-std=c++14"], cc=cc_compiler)
    return ll_code
