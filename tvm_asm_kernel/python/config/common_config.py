import os
from tvm import autotvm

cc_compiler = os.environ["TVM_CC"]

measure_option = autotvm.measure_option(
        # builder=autotvm.LocalBuilder(n_parallel=1, timeout=100), runner=autotvm.LocalRunner(number=1, repeat=20, timeout=300, enable_cpu_cache_flush=True)
        builder=autotvm.LocalBuilder(n_parallel=None, timeout=100), runner=autotvm.LocalRunner(number=100, repeat=1, timeout=300, min_repeat_ms=100)
    )