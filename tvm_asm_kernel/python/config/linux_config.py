import os
from tvm import autotvm

target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon"
