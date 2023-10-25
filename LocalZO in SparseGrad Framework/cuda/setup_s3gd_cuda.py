import sysconfig
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='s3gd_cuda',
    ext_modules=[
        CUDAExtension('s3gd_cuda', [
            's3gd_cuda.cpp',
            's3gd_cuda_kernel.cu',
        ],
        #),
        extra_compile_args={'cxx': [], 'nvcc': ['-ccbin=/home/huanxiong/miniconda3/envs/try/bin/x86_64-conda_cos6-linux-gnu-gcc']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
