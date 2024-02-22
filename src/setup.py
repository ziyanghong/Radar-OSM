from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np




# setup(
#     ext_modules=cythonize("cythonized.pyx",  annotate=True),
# )

ext_modules = [
    Extension(
        "cythonized",
        ["cythonized.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules=cythonize(ext_modules,  annotate=True),
)