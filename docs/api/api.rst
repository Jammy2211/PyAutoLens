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

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   SimulatorImaging
   SimulatorInterferometer

-------
Fitting
-------

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   FitImaging
   FitInterferometer

--------------
Light Profiles
--------------

.. currentmodule:: autogalaxy.profiles.light_profiles

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

.. currentmodule:: autogalaxy.profiles.mass_profiles

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

.. currentmodule:: autoarray.operators.inversion.pixelizations

**Pixelizations:**

.. autosummary::
   :toctree: generated/

   Rectangular
   VoronoiMagnification
   VoronoiBrightnessImage

**Regularizations:**

.. currentmodule:: autoarray.operators.inversion.regularization

.. autosummary::
   :toctree: generated/

   Constant
   AdaptiveBrightness

**Inversions:**

.. currentmodule:: autolens

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

**Pipelines:**

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   PipelineDataset

**Setup:**

.. autosummary::
   :toctree: generated/

   PipelineSetup

.. currentmodule:: autolens.slam

**SLaM:**

.. currentmodule:: autolens.slam

.. autosummary::
   :toctree: generated/

   SLaM
   HyperSetup
   SourceSetup
   LightSetup
   MassSetup