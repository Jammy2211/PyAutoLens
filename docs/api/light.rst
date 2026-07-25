==============
Light Profiles
==============

Standard [``ag.lp``]
--------------------

Standard parametric light profiles whose ``intensity`` is a free parameter of the model.

.. currentmodule:: autogalaxy.profiles.light.standard

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Gaussian
   GaussianSph
   GaussianMultipole
   Sersic
   SersicSph
   SersicMultipole
   Exponential
   ExponentialSph
   DevVaucouleurs
   DevVaucouleursSph
   SersicCore
   SersicCoreSph
   ExponentialCore
   ExponentialCoreSph
   Chameleon
   ChameleonSph
   ElsonFreeFall
   ElsonFreeFallSph

Linear [``ag.lp_linear``]
--------------------------

Linear light profiles whose ``intensity`` is not a free parameter but is instead solved
analytically via a linear matrix inversion during each likelihood evaluation.  This allows
many profiles to be combined efficiently without exploding the non-linear parameter space.

.. currentmodule:: autogalaxy.profiles.light.linear

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Gaussian
   GaussianSph
   GaussianMultipole
   Sersic
   SersicSph
   SersicMultipole
   Exponential
   ExponentialSph
   DevVaucouleurs
   DevVaucouleursSph
   SersicCore
   SersicCoreSph
   ExponentialCore
   ExponentialCoreSph

Operated [``ag.lp_operated``]
------------------------------

Operated light profiles that represent emission which has already had an instrument
operation (e.g. PSF convolution) applied to it.  The ``operated_only`` parameter on
fitting classes controls whether these profiles are included or excluded from a given
image computation.

.. currentmodule:: autogalaxy.profiles.light.operated

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Gaussian
   Moffat
   Sersic

Basis [``ag.lp_basis``]
------------------------

A ``Basis`` groups a collection of light profiles (e.g. a Multi-Gaussian Expansion or
shapelet decomposition) so that they behave as a single profile in the model.  When the
constituent profiles are ``LightProfileLinear`` instances their intensities are all solved
simultaneously via a single inversion.

.. currentmodule:: autogalaxy.profiles.basis

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Basis