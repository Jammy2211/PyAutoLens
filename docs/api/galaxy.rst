===============
Galaxy / Tracer
===============

Galaxy / Tracer
---------------

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