from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy

ext = Extension("tilec.cneedlet", ["tilec/cneedlet.pyx"],
    include_dirs = [numpy.get_include()])

setup(name='tilec',
      version='0.1',
      description='ILC in Tiles',
      url='https://github.com/ACTCollaboration/tilec',
      author='Mathew Madhavacheril',
      author_email='mathewsyriac@gmail.com',
      license='BSD-2-Clause',
      ext_modules = [ext],
      cmdclass = {'build_ext': build_ext},
      packages=['tilec'],
      package_dir={'tilec':'tilec'},
      zip_safe=False)
