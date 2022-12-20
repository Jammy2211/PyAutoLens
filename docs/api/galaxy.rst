=======================
Galaxy / Plane / Tracer
=======================

Galaxy / Plane / Tracer
-----------------------

.. currentmodule:: autolens

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Galaxy
   Plane
   Tracer
   SettingsLens

To treat the redshift of a galaxy as a free parameter in a model, the ``Redshift`` object must
be used.

This is because **PyAutoFit** (which handles model-fitting), requires all parameters to be a Python class.

The ``Redshift`` object does not need to be used for general **PyAutoGalaxy** use.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

    Redshift

By using a ``HyperGalaxy``, the noise-map value in the regions of the image that the galaxy is located
are increased.

This prevents over-fitting regions of the data where the model does not provide a good fit
(e.g. where a high chi-squared is inferred).

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   HyperGalaxy