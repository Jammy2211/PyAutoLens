===============
Galaxy / Tracer
===============

Galaxy / Tracer
---------------

``Galaxy`` and ``Galaxies`` model individual galaxies with light and mass profiles at a
given redshift.  ``Tracer`` groups galaxies by redshift into planes and performs
multi-plane gravitational lensing ray-tracing, computing lensed images, convergence,
deflection angles, magnification, critical curves, and caustics.

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Galaxy
   Galaxies
   Tracer

To treat the redshift of a galaxy as a free parameter in a model, the ``Redshift`` object must
be used.

This is because **PyAutoFit** (which handles model-fitting), requires all parameters to be a Python class.

The ``Redshift`` object does not need to be used for general **PyAutoGalaxy** use.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

    Redshift