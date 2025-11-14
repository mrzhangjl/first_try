from setuptools import setup,Extension
from Cython.Build import cythonize
ext = Extension(
    name = "cython_add_module",
    sources = ["cython_add.pyx"]
)
setup(
    name="Cython Add Module",
    ext_modules=cythonize(ext)
)