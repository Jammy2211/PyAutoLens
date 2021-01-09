=============
API Reference
=============

---------------
Data Structures
---------------

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   Array
   ValuesIrregularGrouped
   Grid
   GridIterate
   GridInterpolate
   GridIrregularGrouped
   Kernel
   Convolver
   Visibilities
   TransformerDFT
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

.. currentmodule:: autoarray.inversion.pixelizations

**Pixelizations:**

.. autosummary::
   :toctree: generated/

   Rectangular
   VoronoiMagnification
   VoronoiBrightnessImage

.. currentmodule:: autoarray.inversion.regularization

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


-----
Plots
-----

.. currentmodule:: autolens.plot

**Plotters**

.. autosummary::
   :toctree: generated/

   Plotter
   Plotter
   Include

**Matplotlib Objects**

.. autosummary::
   :toctree: generated/

   Figure
   Cmap
   Colorbar
   Ticks
   Labels
   Legend
   Units
   Output
   OriginScatter
   MaskScatter
   BorderScatter
   GridScatter
   PositionsScatter
   IndexScatter
   PixelizationGridScatter
   Line
   ArrayOverlayer
   VoronoiDrawer
   LightProfileCentreScatter
   MassProfileCentreScatter
   MultipleImagesScatter
   CriticalCurvesPlot
   CausticsPlot

**Plots:**

.. autosummary::
   :toctree: generated/

   Array
   Grid
   Line
   MapperObj
   Imaging
   Interferometer
   MassProfile
   LightProfile
   Galaxy
   Plane
   Tracer
   FitImaging
   FitInterferometer
   FitGalaxy
   Inversion
   Mapper

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

   SettingsPhaseImaging
   SettingsPhaseInterferometer

**Pipelines:**

.. autosummary::
   :toctree: generated/

   PipelineDataset

**Setup:**

.. autosummary::
   :toctree: generated/

    SetupHyper
    SetupLightParametric
    SetupLightParametric
    SetupMassTotal
    SetupMassLightDark
    SetupSourceParametric
    SetupSourceInversion
    SetupSMBH
    SetupSubhalo
    SetupPipeline

**SLaM:**

.. autosummary::
   :toctree: generated/

   SLaM
   SLaMPipelineSourceParametric
   SLaMPipelineSourceInversion
   SLaMPipelineLightParametric
   SLaMPipelineMass

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