from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

setup(
  name = 'cpu_nums',
  ext_modules = cythonize([
      Extension("utils.cpu_nms",["cpu_nms.pyx"],
      include_dirs = ["C:\\Users\\lewis\\Anaconda3\\Lib\\site-packages\\numpy\\core\\include"],
      libraries = ["npymath"],
      library_dirs = ["C:\\Users\\lewis\\Anaconda3\\Lib\\site-packages\\numpy\\core\\lib"]),
      ]),
)
