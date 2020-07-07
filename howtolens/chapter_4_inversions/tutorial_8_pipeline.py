import autofit as af
import autolens as al

"""
In this pipeline, we fit the a strong lens using a _EllipticalIsothermal_ mass profile and a source which uses an
inversion.

The pipeline is three phases:

Phase 1:

    Fit the lens mass model and source _LightProfile_.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: None.
    Notes: None.

Phase 2:

    Fit the source inversion using the lens _MassProfile_ inferred in phase 1.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens & Mass (instance -> phase1).
    Notes: Lens mass fixed, source inversion parameters vary.

Phase 3:

    Refines the lens light and mass models using the source inversion of phase 2.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Mass (model -> phase 1), Source Inversion (instance -> phase 2)
    Notes: Lens mass varies, source inversion parameters fixed.
"""


def make_pipeline(setup, settings, folders=None):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__inversion"

    """
    This pipeline is tagged according to whether:

        1) The lens galaxy mass model includes an external shear.
        2) The pixelization and regularization scheme of the pipeline (fitted in phases 3 & 4).
    """

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50, evidence_tolerance=5.0),
    )

    phase1.search.facc = 0.3
    phase1.search.const_efficiency_mode = True

    """
    Phase 2: Fit the input pipeline pixelization & regularization, where we:

        1) Set lens's mass model using the results of phase 1.
    """

    source = al.GalaxyModel(
        redshift=1.0,
        pixelization=al.pix.VoronoiMagnification,
        regularization=al.reg.Constant,
    )

    """We can customize the inversion's priors like we do our light and mass profiles."""

    source.pixelization.shape_0 = af.UniformPrior(lower_limit=20.0, upper_limit=40.0)
    source.pixelization.shape_1 = af.UniformPrior(lower_limit=20.0, upper_limit=40.0)

    """
    The expected value of the regularization_coefficient depends on the details of the dataset reduction and
    source galaxy. A broad log-uniform prior is thus an appropriate way to sample the large range of
    possible values.
    """

    source.regularization.coefficient = af.LogUniformPrior(
        lower_limit=1.0e-6, upper_limit=10000.0
    )

    phase2 = al.PhaseImaging(
        phase_name="phase_2__source_inversion_initialize",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=source,
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    """
    We now 'extend' phase 1 with an additional 'inversion phase' which uses the maximum log likelihood mass model of 
    phase 1 above to refine the _Inversion_, by fitting only the parameters of the _Pixelization_ and _Regularization_
    (in this case, the shape of the _VoronoiMagnification_ and regularization coefficient of the _Constant_.

    The the _Inversion_ phase results are accessible as attributes of the phase results and used in phase 3 below.
    """

    phase2 = phase2.extend_with_inversion_phase(
        inversion_search=af.DynestyStatic(n_live_points=50)
    )

    """
    Phase 3: Fit the lens's mass using the input pipeline pixelization & regularization, where we:

        1) Fix the source inversion parameters to the results of the extended inversion phase of phase 2.
        2) Set priors on the lens galaxy mass using the results of phase 1.
    """

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sie__source_inversion",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase2.results.inversion.instance.galaxies.source.pixelization,
                regularization=phase2.results.inversion.instance.galaxies.source.regularization,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3)
