import os
import torch
from setuptools import setup
from torch.utils import cpp_extension

sources = ["src/backend.cpp", "src/allgather.cu", "src/allreduce.cu", "src/reduce_scatter.cu"]
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/src/"]

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name="mini_nccl",
        sources=sources,
        include_dirs=include_dirs,
    )
else:
    raise ValueError("Please install torch with CUDA backend to run mini-nccl.")

setup(
    name="MiniNCCL",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)