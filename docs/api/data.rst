===============
Data Structures
===============


2D Data Structures
------------------

Two-dimensional data structures store and mask 2D arrays containing data (e.g. images) and
grids of (y,x) Cartesian coordinates (which are used for evaluating light profiles).

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Mask2D
   Array2D
   Grid2D
   Grid2DIterate
   Grid2DIrregular

Imaging
-------

For datasets taken with a CCD (or similar imaging device), including objects which perform
2D convolution.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Imaging
   SettingsImaging
   SimulatorImaging
   Kernel2D
   Convolver


Interferometer
--------------

For datasets taken with an interferometer (E.g. ALMA), including objects which perform
a fast Fourier transform to map data to the uv-plane.

.. autosummary::
   :toctree: _autosummary

   Interferometer
   SettingsInterferometer
   SimulatorInterferometer
   Visibilities
   TransformerDFT
   TransformerNUFFT

1D Data Structures
------------------

One-dimensional data structures store and mask 1D arrays and grids of (x) Cartesian
coordinates.

Their most common use is manipulating 1D representations of a light or mass
profile (e.g. computing the intensity versus radius in 1D, or convergene vs radius).

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Mask1D
   Array1D
   ValuesIrregular
   Grid1D