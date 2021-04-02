=============
API Reference
=============

---------------
Data Structures
---------------

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   Array1D
   Array2D
   ValuesIrregular
   Grid1D
   Grid2D
   Grid2DIterate
   Grid2DInterpolate
   Grid2DIrregular
   Kernel2D
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
   Interferometer

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

   EllGaussian
   SphGaussian
   EllSersic
   SphSersic
   EllExponential
   SphExponential
   EllDevVaucouleurs
   SphDevVaucouleurs
   EllSersicCore
   SphSersicCore

-------------
Mass Profiles
-------------

.. currentmodule:: autogalaxy.profiles.mass_profiles

**Total Mass Profiles:**

.. autosummary::
   :toctree: generated/

    PointMass
    EllPowerLawCored
    SphPowerLawCored
    EllPowerLawBroken
    SphPowerLawBroken
    EllIsothermalCored
    SphIsothermalCored
    EllPowerLaw
    SphPowerLaw
    EllIsothermal
    SphIsothermal

**Dark Mass Profiles:**

.. autosummary::
   :toctree: generated/

    EllNFWGeneralized
    SphNFWGeneralized
    SphNFWTruncated
    SphNFWTruncatedMCRDuffy
    SphNFWTruncatedMCRLudlow
    EllNFW
    SphNFW
    SphNFWMCRDuffy
    SphNFWMCRLudlow

**Stellar Mass Profiles:**

.. autosummary::
   :toctree: generated/

    EllGaussian
    EllSersic
    SphSersic
    EllExponential
    SphExponential
    EllDevVaucouleurs
    SphDevVaucouleurs
    EllSersicRadialGradient
    SphSersicRadialGradient

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

   Array2D
   Grid2D
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

**Setup:**

.. autosummary::
   :toctree: generated/

    SetupHyper

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