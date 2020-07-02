=================
API Documentation
=================

---------------
Data Structures
---------------

.. currentmodule:: autoarray

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

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   Imaging
   MaskedImaging
   Interferometer
   MaskedInterferometer

----------
Simulators
----------

.. autosummary::
   :toctree: generated/

   SimulatorImaging
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

.. autosummary::
   :toctree: generated/

   Galaxy
   Plane
   Tracer

----------
Inversions
----------

**Pixelizations:**

.. autosummary::
   :toctree: generated/

   Rectangular
   VoronoiMagnification
   VoronoiBrightnessImage

**Regularizations:**

.. autosummary::
   :toctree: generated/

   Constant
   AdaptiveBrightness

**Inversions:**

.. autosummary::
   :toctree: generated/

   Mapper
   Inversion

-------------
Lens Modeling
-------------

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

**Pipelines:**

.. autosummary::
   :toctree: generated/

   PipelineDataset

**Setup:**

.. autosummary::
   :toctree: generated/

   PipelineSetup

**SLaM:**

.. autosummary::
   :toctree: generated/

   SLaM
   HyperSetup
   SourceSetup
   LightSetup
   MassSetup

---------
PyAutoFit
---------

**Searches:**

.. currentmodule:: autofit

.. autosummary::
   :toctree: generated/

   DynestyStatic
   DynestyDynamic
   Emcee
   PySwarmsGlobal
   MultiNest