=============
API Reference
=============

---------------
Data Structures
---------------

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   Mask1D
   Mask2D
   Array1D
   Array2D
   ValuesIrregular
   Grid1D
   Grid2D
   Grid2DIterate
   Grid2DIrregular
   Kernel2D
   Convolver
   Visibilities
   TransformerDFT
   TransformerNUFFT

-------
Imaging
-------

.. currentmodule:: autolens

.. autosummary::
   :toctree: generated/

   Imaging
   SettingsImaging
   SimulatorImaging
   FitImaging
   AnalysisImaging

--------------
Interferometer
--------------

.. autosummary::
   :toctree: generated/

   SettingsInterferometer
   Interferometer
   SimulatorInterferometer
   FitInterferometer
   AnalysisInterferometer

---------------------
Point Source Modeling
---------------------

.. autosummary::
   :toctree: generated/

   PointDataset
   PointDict
   FitPositionsImage
   FitFluxes
   PointSolver
   AnalysisPoint

-------------
Lens Modeling
-------------

.. currentmodule:: autolens

**Setup:**

.. autosummary::
   :toctree: generated/

    SetupHyper
    Preloads

**Searches:**

.. currentmodule:: autofit

.. autosummary::
   :toctree: generated/

   DynestyStatic
   DynestyDynamic
   Emcee
   PySwarmsLocal
   PySwarmsGlobal


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
   EllExponentialCore
   SphExponentialCore
   EllChameleon
   SphChameleon
   EllEff
   SphEff

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
    SphNFWTruncatedMCRScatterLudlow
    EllNFW
    SphNFW
    SphNFWMCRDuffy
    SphNFWMCRLudlow
    EllNFWMCRScatterLudlow
    SphNFWMCRScatterLudlow
    EllNFWMCRLudlow
    EllNFWGeneralizedMCRLudlow

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
    EllChameleon
    SphChameleon

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
   SettingsLens

----------
Inversions
----------

.. currentmodule:: autoarray.inversion.pixelizations

**Pixelizations:**

.. autosummary::
   :toctree: generated/

   Rectangular
   DelaunayMagnification
   DelaunayBrightnessImage
   VoronoiMagnification
   VoronoiBrightnessImage
   VoronoiNNMagnification
   VoronoiNNBrightnessImage

.. currentmodule:: autoarray.inversion.regularization

**Regularizations:**

.. autosummary::
   :toctree: generated/

   Constant
   ConstantSplit
   AdaptiveBrightness
   AdaptiveBrightnessSplit

.. currentmodule:: autolens

**LEqs:**

.. autosummary::
   :toctree: generated/

   Mapper

**Settings:**

.. autosummary::
   :toctree: generated/

   SettingsPixelization
   SettingsInversion

-----
Plots
-----

.. currentmodule:: autolens.plot

**Plotters:**

.. autosummary::
   :toctree: generated/

    Array2DPlotter
    Grid2DPlotter
    MapperPlotter
    YX1DPlotter
    InversionPlotter
    ImagingPlotter
    InterferometerPlotter
    LightProfilePlotter
    LightProfilePDFPlotter
    MassProfilePlotter
    MassProfilePDFPlotter
    GalaxyPlotter
    FitImagingPlotter
    FitInterferometerPlotter
    PlanePlotter
    HyperPlotter
    FitImagingPlotter
    FitInterferometerPlotter
    TracerPlotter
    MultiFigurePlotter
    MultiYX1DPlotter

**Search Plotters:**

.. autosummary::
   :toctree: generated/

   DynestyPlotter
   UltraNestPlotter
   EmceePlotter
   ZeusPlotter
   PySwarmsPlotter

**Plot Customization Objects**

.. autosummary::
   :toctree: generated/

    MatPlot1D
    MatPlot2D
    Include1D
    Include2D
    Visuals1D
    Visuals2D

**Matplotlib Wrapper Base Objects:**

.. autosummary::
   :toctree: generated/

    Units
    Figure
    Axis
    Cmap
    Colorbar
    ColorbarTickParams
    TickParams
    YTicks
    XTicks
    Title
    YLabel
    XLabel
    Legend
    Output

**Matplotlib Wrapper 1D Objects:**

.. autosummary::
   :toctree: generated/

    YXPlot

**Matplotlib Wrapper 2D Objects:**

.. autosummary::
   :toctree: generated/

    ArrayOverlay
    GridScatter
    GridPlot
    VectorYXQuiver
    PatchOverlay
    VoronoiDrawer
    OriginScatter
    MaskScatter
    BorderScatter
    PositionsScatter
    IndexScatter
    PixelizationGridScatter
    ParallelOverscanPlot
    SerialPrescanPlot
    SerialOverscanPlot
