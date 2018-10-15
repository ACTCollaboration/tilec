from distutils.core import setup, Extension
import os



setup(name='tilec',
      version='0.1',
      description='ILC in Tiles',
      url='https://github.com/ACTCollaboration/tilec',
      author='Mathew Madhavacheril',
      author_email='mathewsyriac@gmail.com',
      license='BSD-2-Clause',
      packages=['tilec'],
      package_dir={'tilec':'tilec'},
      zip_safe=False)
