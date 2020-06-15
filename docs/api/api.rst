=================
API Documentation
=================

---------------
Data Structures
---------------

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   Array
   Values
   Grid
   GridIterate
   GridInterpolate
   GridCoordinates
   Kernel
   Convolver
   Visibilities
   TransformerDFT
   TransformerFFT
   TransformerNUFFT

--------
Datasets
--------

.. autosummary::
   :toctree: generated/

   Imaging
   MaskedImaging
   SimulatorImaging
   Interferometer
   MaskedInterferometer
   SimulatorInterferometer

-------
Fitting
-------

.. autosummary::
   :toctree: generated/

   FitImaging
   FitInterferometer

--------------
Light Profiles
--------------

.. currentmodule:: autolens.lp

.. autosummary::
   :toctree: generated/

   EllipticalGaussian
   SphericalGaussian
   EllipticalSersic
   SphericalSersic
   EllipticalExponential
   SphericalExponential
   EllipticalDevVaucouleurs
   SphericalDevVaucouleurs
   EllipticalCoreSersic
   SphericalCoreSersic

-------------
Mass Profiles
-------------

.. currentmodule:: autolens.mp

**Total Mass Profiles:**

.. autosummary::
   :toctree: generated/

    PointMass
    EllipticalCoredPowerLaw
    SphericalCoredPowerLaw
    EllipticalBrokenPowerLaw
    SphericalBrokenPowerLaw
    EllipticalCoredIsothermal
    SphericalCoredIsothermal
    EllipticalPowerLaw
    SphericalPowerLaw
    EllipticalIsothermal
    SphericalIsothermal

**Dark Mass Profiles:**

.. autosummary::
   :toctree: generated/

    EllipticalGeneralizedNFW
    SphericalGeneralizedNFW
    SphericalTruncatedNFW
    SphericalTruncatedNFWMCRDuffy
    SphericalTruncatedNFWMCRLudlow
    SphericalTruncatedNFWMCRChallenge
    EllipticalNFW
    SphericalNFW
    SphericalNFWMCRDuffy
    SphericalNFWMCRLudlow

**Stellar Mass Profiles:**

.. autosummary::
   :toctree: generated/

    EllipticalGaussian
    EllipticalSersic
    SphericalSersic
    EllipticalExponential
    SphericalExponential
    EllipticalDevVaucouleurs
    SphericalDevVaucouleurs
    EllipticalSersicRadialGradient
    SphericalSersicRadialGradient

**Mass-sheets:**

.. autosummary::
   :toctree: generated/

   ExternalShear
   MassSheet

-------
Lensing
-------

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   Galaxy
   Plane
   Tracer

----------
Inversions
----------

.. currentmodule:: autolens.pix

**Pixelizations:**

.. autosummary::
   :toctree: generated/

   Rectangular
   VoronoiMagnification
   VoronoiBrightnessImage

.. currentmodule:: autolens.reg

**Regularizations:**

.. autosummary::
   :toctree: generated/

   Constant
   AdaptiveBrightness

.. currentmodule:: autolens

**Inversions:**

.. autosummary::
   :toctree: generated/

   Mapper
   Inversion

-------------
Lens Modeling
-------------

.. currentmodule:: autolens

**Phases:**

.. autosummary::
   :toctree: generated/

   PhaseImaging
   PhaseInterferometer

**Settings:**

.. autosummary::
   :toctree: generated/

   PhaseSettingsImaging
   PhaseSettingsInterferometer

**Searches:**

.. currentmodule:: autofit

.. autosummary::
   :toctree: generated/

   DynestyStatic
   DynestyDynamic
   Emcee
   PySwarmsGlobal
 MultiNest

.. currentmodule:: autolens

**Pipelines:**

.. autosummary::
   :toctree: generated/

   PipelineDataset

**Setup:**

.. autosummary::
   :toctree: generated/

   PipelineSetup

.. currentmodule:: autolens.slam

**SLaM:**

.. autosummary::
   :toctree: generated/

   SLaM
   Hyper
   Source
   Light
   Mass