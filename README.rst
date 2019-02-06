=======
tILe-C
=======

`tILe-C` is ILC in tiles. It is both a library for CMB foregrounds and harmonic
ILC as well as a set of pipeline scripts designed primarily for component
separation of high-resolution ground-based CMB maps that might have
inhomogenous, anisotropic and inhomogenously anisotropic noise.

* Free software: BSD license
* Documentation: in the works

Dependencies
------------

* Python>=2.7 or Python>=3.4
* numpy, scipy, matplotlib
* pixell

Development and Contributing
----------------------------

This code is currently being developed by Mathew Madhavacheril, Colin Hill and
Sigurd Naess. Contributions are welcome and encouraged through pull requests.


Installation
------------

Clone this repository and install with symbolic links as follows
so that changes you make to the code are immediately reflected.


.. code-block:: console

   pip install -e . --user


Pointing to data
----------------

You can start running the pipeline scripts as soon as you set up a configuration
file that tells these scripts where relevant data products are stored. To do
this, copy the template in `input/paths_template.yml` to `paths.yml`. The former
is version controlled and the latter is not. Edit the latter to conform to where
you have stored the relevant data on the system you have just cloned this
repository to.

The data model itself is implemented in `tilec/datamodel.py`. To interface with
the data products of a new experiment (or version), you are encouraged to modify
that module and submit your changes as a pull request to be incorporated into
the master branch here. Modifications are preferred to be in the form of
subclasses of `tilec.datamodel.DataModel`. See the ACT and Planck examples
there.

You can then add descriptions of ``arrays'' to `data.yml' that point to
implemented data models. The overall analysis is tied to a `mask' whose
WCS geometry specifies what pixel regions of the maps in each array
are loaded and coadded together. The mask is also assumed to be apodized
and is applied right before any FFT is performed.


Running ILC
-----------

1. Covariance
   ~~~~~~~~~~

The first step is to generate an empirical covariance matrix. The pipeline
script for this is ``bin/cov.py''. The call options are:

.. code-block:: console

   python bin/cov.py <version> <region>
   <comma,separated,list,of,array,short,names>
   [--signal-bin-width I] [--signal-interp-order I] [--dfact I]
   [--rfit-bin-width I] [--rfit-wnoise-width I] [--rfit-lmin I]


An example call is:


.. code-block:: console

   python bin/cov.py v0.1 deep56 a1,a2,p1,p2

This will make a directory named v0.1 in the default save location, a
sub-directory named deep56, use masks corresponding to deep56, generate
the non-redundant parts of a TILe-C hybrid covariance matrix involving
arrays with short names a1,a2,p1,p2 and save it to that sub-directory along
with a copy of the configuration options.

2. Sub-covariance
   ~~~~~~~~~~~~~~

If you want to slice an existing covariance, use the bin/slicecov.py script.

.. code-block:: console

   python bin/slicecov.py <version_full> <version_sliced>
   <comma,separated,list,of,array,short,names>



