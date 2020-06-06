=======
tILe-C
=======

.. image:: https://travis-ci.org/simonsobs/tilec.svg?branch=master
           :target: https://travis-ci.org/ACTCollaboration/tilec

.. image:: https://coveralls.io/repos/github/ACTCollaboration/tilec/badge.svg?branch=master
		   :target: https://coveralls.io/github/ACTCollaboration/tilec?branch=master



``tILe-C`` is ILC in tiles. It is both a library for CMB foregrounds and harmonic
ILC as well as a set of pipeline scripts designed primarily for component
separation of high-resolution ground-based CMB maps that might have
inhomogenous, anisotropic and inhomogenously anisotropic noise. This code was
used to make the products presented in Madhavacheril, Hill, Naess at. al. 2019
(MHN19, arxiv_).

* Free software: BSD license

Dependencies
------------

* Python>=3.6
* numpy, scipy, matplotlib
* orphics_ (can be git cloned and installed with ``pip install -e . --user``)
* pixell_, soapack_

Development and Contributing
----------------------------

This code is maintained by Mathew Madhavacheril, Colin Hill and
Sigurd Naess. Contributions are welcome and encouraged through pull requests.


Installation
------------

The main non-trivial dependencies are pixell_ and soapack_. The latter must be
set up with a config file that points to relevant data directories for the input
maps and beams. See the instructions in soapack README.

Following this, edit your ~/.soapack.yml to include a section named ``tilec`` that
has keywords ``save_path`` and ``scratch_path`` that point to new directories where
you would like to store tilec related files. The former is where the ILC
products are stored and the latter is where temporary files are stored.


Once those are set up, clone this repository and install with symbolic links as follows
so that changes you make to the code are immediately reflected.


.. code-block:: console

				pip install -e . --user


Running ILC
-----------

1. Covariance in 2D Fourier space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to generate an empirical covariance matrix. The pipeline
script for this is ``bin/make_cov.py``. The command line arguments and their
descriptions can be obtained by running ``python bin/make_cov.py -h``.

The following commands were run to produce the final versions of maps in
MHN19:

.. code-block:: console

				python bin/make_cov.py v1.2.0 deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 -o
				python bin/make_cov.py v1.2.0 boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 --o

This will make a directory in the default save location, use masks corresponding
to deep56 and boss, generate the non-redundant parts of a TILe-C hybrid
covariance matrix and save these to disk.

2. Linear Co-Addition in 2D Fourier space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second step is to derive ILC weights from the above covariance matrices and
responses to desired components and use these to co-add the input arrays in 2D
Fourier space. The pipeline
script for this is ``bin/make_ilc.py``. The command line arguments and their
descriptions can be obtained by running ``python bin/make_ilc.py -h``.

The following commands were run to produce the final versions of maps in
MHN19:

.. code-block:: console

				python bin/make_ilc.py map_v1.2.0_joint v1.2.0 deep56 d56_01,d56_02,d56_03,d56_04,d56_05,d56_06,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,CMB-CIB,tSZ-CMB,tSZ-CIB 1.6,1.6,2.4,2.4,2.4,2.4
				python bin/make_ilc.py map_v1.2.0_joint v1.2.0 boss boss_01,boss_02,boss_03,boss_04,p01,p02,p03,p04,p05,p06,p07,p08 CMB,tSZ,CMB-tSZ,tSZ-CMB,tSZ-CIB,CMB-CIB 1.6,1.6,2.4,2.4,2.4,2.4



.. _pixell: https://github.com/simonsobs/pixell/
.. _orphics: https://github.com/msyriac/orphics/
.. _soapack: https://github.com/simonsobs/soapack/
.. _arxiv: https://arxiv.org/abs/1911.05717

