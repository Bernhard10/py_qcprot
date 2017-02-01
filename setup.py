from distutils.core import setup, Extension
from Cython.Build import cythonize

ext_modules=[
    Extension("py_qcprot",
              sources=["py_qcprot.pyx", "qcprot.c"],
    )
]
setup(
  name = 'py_qcprot',
  ext_modules = cythonize(ext_modules)
)
