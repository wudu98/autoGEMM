import os
from tvm import autotvm

target = "llvm -mtriple=arm64-apple-darwin -mattr=+neon"
