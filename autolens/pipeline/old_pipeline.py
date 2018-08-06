from autolens.analysis import analysis as an
from autolens.imaging import masked_image as mi
from autolens.analysis import galaxy_prior
from autolens.profiles import light_profiles, mass_profiles
from autolens.autopipe import model_mapper
from autolens.autopipe import non_linear
from autolens.pipeline import phase as ph
from autolens.pixelization import pixelization
from autolens.analysis import galaxy

import os
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)


def source_only_pipeline(image, mask):
    """
    Fit a source-only image (i.e. no lens light component)

    Parameters
    ----------
    image: Image
        An image of a lens galaxy including metadata such as PSF, background noise and effective exposure time
    mask: Mask
        A mask describing which parts of the image should be excluded from the analysis

    Returns
    -------
    result: [Result]
        An array of results, one for each stage of the analysis
    """

    logger.info(
        """
        Pipeline 1:

        PURPOSE - Fit a source-only image (i.e. no lens light component)

        PREPROCESSING:

        - Mark the brightest regions / multiple images of the source.
        - Draw a circle tracing the source (Einstein Radius / centre)
        - Draw circle / ellipse for the mask.

        NOTES:

        Image: Observed image used throughout.
        Mask: Assume a large mask (e.g. 2") throughout - this value could be chosen in preprocessing.
        """
    )

    # Create an array in which to store results
    results = []

    # Create an analysis object for the image and mask
    masked_image = mi.MaskedImage(image, mask)

    logger.info(
        """
        1) Mass: SIE+Shear
           Source: Sersic
           NLO: LM
        """
    )

    # Create an optimizer
    phase1 = ph.ParameterizedPhase(optimizer=non_linear.MultiNest())

    # Define galaxy priors
    source_galaxy_prior = galaxy_prior.GalaxyPrior(light_profile=light_profiles.EllipticalSersic)
    lens_galaxy_prior = galaxy_prior.GalaxyPrior(spherical_mass_profile=mass_profiles.EllipticalIsothermal,
                                                 shear_mass_profile=mass_profiles.ExternalShear)

    # Add the galaxy priors to the optimizer
    phase1.add_variable_lens_galaxies([lens_galaxy_prior])
    phase1.add_variable_source_galaxies([source_galaxy_prior])

    # Analyse the system
    result_1 = phase1.fit(masked_image)

    # Add the result of the first analysis to the list
    results.append(result_1)

    logger.info(
        """
        2) Mass: SIE+Shear (priors from phase 1)
           Source: 'smooth' pixelization (include regularization parameter(s) in the model)
           NLO: LM
        """
    )
    # Create an optimizer
    phase = ph.ParameterizedPhase(optimizer=non_linear.MultiNest())
    optimizer_2 = non_linear.DownhillSimplex()

    # Define galaxy priors
    lens_galaxy_prior = galaxy_prior.GalaxyPrior(spherical_mass_profile=mass_profiles.EllipticalIsothermal,
                                                 shear_mass_profile=mass_profiles.ExternalShear)
    # The source galaxy is now represented by a pixelization
    source_galaxy_prior = galaxy_prior.GalaxyPrior(pixelization=pixelization.Rectangular)

    # Add the galaxy priors to the optimizer
    optimizer_2.variable.lens_galaxies = [lens_galaxy_prior]
    optimizer_2.variable.source_galaxies = [source_galaxy_prior]

    # Associate priors found in the first analysis with the new galaxy priors
    lens_galaxy_prior_result = result_1.priors.lens_galaxies[0]
    lens_galaxy_prior.spherical_mass_profile = lens_galaxy_prior_result.spherical_mass_profile
    lens_galaxy_prior.shear_mass_profile = lens_galaxy_prior_result.shear_mass_profile

    # Analyse the system
    result_2 = optimizer_2.fit(analysis)

    # Add the result of the second analysis to the list
    results.append(result_2)

    logger.info(
        """
        2H) Hyper-parameters: All included in model (most priors broad and uniform, but use previous phase 
            regularization as well)
            Mass: SIE+Shear (Fixed to highest likelihood model from phase 2)
            Source: 'noisy' pixelization
            NLO: MN
        """
    )
    # Create an optimizer
    optimizer_2h = non_linear.MultiNest()

    # Define a single galaxy prior that is a pixelized galaxy
    source_galaxy_prior = galaxy_prior.GalaxyPrior(pixelization=pixelization.Voronoi)

    #  Add the variable pixelization to the optimizer
    optimizer_2h.variable.source_galaxies = [source_galaxy_prior]

    # Set the regularization prior using results from analysis 2
    source_galaxy_prior.pixelization.regularization = result_2.priors.pixelization.regularization

    # Set a constant lens galaxy from the previous result
    optimizer_2h.constant.lens_galaxies = result_2.instance.lens_galaxies

    # Analyse the system
    result_2h = optimizer_2h.fit(analysis)

    # Add the result of analysis 2h to the results
    results.append(result_2h)

    logger.info(
        """
        a) Mass: SPLE+Shear (priors from Init phase 2)
           Source: 'noisy' pixelization (Fixed to init 2H hyper-parameters)
        """
    )
    # Create an optimizer
    optimizer_a = non_linear.MultiNest()

    # Define a lens galaxy prior
    lens_galaxy_prior = galaxy_prior.GalaxyPrior(spherical_power_law_mass_profile=mass_profiles.SphericalPowerLaw,
                                                 shear_mass_profile=mass_profiles.ExternalShear)

    # Add the lens galaxy prior to the optimizer
    optimizer_a.variable.lens_galaxies = [lens_galaxy_prior]

    # Set some lens galaxy priors using results from analysis 2
    lens_galaxy_prior_result = result_2.priors.lens_galaxies[0]
    lens_galaxy_prior.shear_mass_profile = lens_galaxy_prior_result.shear_mass_profile
    lens_galaxy_prior.spherical_power_law_mass_profile.centre = lens_galaxy_prior_result.spherical_mass_profile.centre

    # Set a source galaxy with a fixed pixelization from analysis 2
    optimizer_a.constant.source_galaxies = result_2h.instance.source_galaxies

    # Analyse the system
    result_a = optimizer_a.fit(analysis)

    # Add the result of the main analysis to the results
    results.append(result_a)

    logger.info("Pipeline complete")

    # Return the results
    return results


def lens_and_source_pipeline(image, lens_mask, source_mask, combined_mask):
    """
    Pipeline 2:

    PURPOSE - Fit a lens light + source image (mass model does not decomposed the light and dark matter)

    PREPROCESSING:

    - Mark the brightest regions / multiple images of the source.
    - Draw a circle around the source (Einstein Radius / center)
    - Mark the centre of the lens light.
    - Draw circle / ellipse for mask containing the lens galaxy.
    - Draw annulus for mask containing the source (excludes central regions of image where lens is).
    """

    # Create an array in which to store results
    results = []
    """

    1) Image: Observed image, includes lens+source.
       Mask: Circle / Ellipse containing entire lens galaxy.
       Light: Sersic (This phase simply subtracts the lens light from the image - the source is present and disrupts the 
       NLO: LM        fit but we choose not to care)
       Purpose: Provides lens subtracted image for subsequent phases.

    """
    # Create an analysis object for the image and a mask that means only lens light is visible
    lens_light_analysis = an.Analysis(image, lens_mask)

    optimizer_1 = non_linear.DownhillSimplex()

    lens_galaxy = galaxy_prior.GalaxyPrior(light_profile=light_profiles.EllipticalSersic)
    lens_galaxy.redshift = model_mapper.Constant(1)

    optimizer_1.variable.lens_galaxies = [lens_galaxy]

    result_1 = optimizer_1.fit(lens_light_analysis)

    results.append(result_1)

    # TODO: simplify this stage of the process
    light_grid = result_1.lens_galaxies[0].light_profile.intensity_at_coordinates(
        lens_light_analysis.coords_collection.image)

    source_image = image - lens_light_analysis.mapper_collection.data_to_pixel.map_to_2d(light_grid)

    """
    2) Image: The lens light subtracted image from phase 1.
       Mask: Annulus mask containing just source
       Light: None
       Mass: SIE+Shear
       Source: Sersic
       NLO: LM
       Purpose: Provides mass model priors for next phase.
    """

    source_analysis = an.Analysis(source_image, source_mask)

    optimizer_2 = non_linear.DownhillSimplex()

    lens_galaxy = galaxy_prior.GalaxyPrior(sie_mass_profile=mass_profiles.SphericalIsothermal,
                                           shear_mass_profile=mass_profiles.ExternalShear)
    source_galaxy = galaxy_prior.GalaxyPrior(light_profile=light_profiles.EllipticalSersic)

    optimizer_2.variable.lens_galaxies = [lens_galaxy]
    optimizer_2.variable.source_galaxies = [source_galaxy]

    result_2 = optimizer_2.fit(source_analysis)

    results.append(result_2)

    """
    3) Image: The lens light subtracted image from phase 1.
       Mask: Circle / Ellipse containing entire lens galaxy.
       Light: None
       Mass: SIE+Shear (priors from phase 2)
       Source: 'smooth' pixelization (include regularization parameter(s) in the model)
       NLO: LM
       Purpose: Refines mass model and sets up the source-plane pixelization regularization.
    """

    pixelized_source_analysis = an.Analysis(source_image, combined_mask)

    optimizer_3 = non_linear.DownhillSimplex()

    source_galaxy = galaxy_prior.GalaxyPrior(pixelization=pixelization.SquarePixelization)
    lens_galaxy = galaxy_prior.GalaxyPrior(sie_mass_profile=mass_profiles.SphericalIsothermal,
                                           shear_mass_profile=mass_profiles.ExternalShear)

    lens_galaxy_result = result_2.priors.lens_galaxies[0]

    lens_galaxy.sie_mass_profile.centre = lens_galaxy_result.sie_mass_profile.centre
    lens_galaxy.sie_mass_profile.einstein_radius = lens_galaxy_result.sie_mass_profile.einstein_radius

    lens_galaxy.shear_mass_profile.magnitude = lens_galaxy_result.shear_mass_profile.magnitude
    lens_galaxy.shear_mass_profile.phi = lens_galaxy_result.shear_mass_profile.phi

    optimizer_3.variable.lens_galaxies = [lens_galaxy]
    optimizer_3.variable.source_galaxies = [source_galaxy]

    result_3 = optimizer_3.fit(pixelized_source_analysis)
    results.append(result_3)

    """
    4) Image: Observed image, includes lens+source.
       Mask: Circle / Ellipse containing entire lens galaxy.
       Light: Sersic + Exponential (shared centre / phi, include Instrumentation lens light noise scaling parameters in 
              model)
       Mass: SIE+Shear (fixed to results from phase 3)
       Source: 'smooth' pixelization (include regularization parameter(s) in the model, using previous phase prior)
       NLO: LM
       Purpose: To fit a complex light profile, we need to do so simultaneously with the source reconstruction to avoid
                systematics. Thus, this rather confusing phase sets it up so that the mass profile is fixed whilst we
                get ourselves a good multi-component light profile.
    """

    full_light_analysis = an.Analysis(image, combined_mask)

    optimizer_4 = non_linear.DownhillSimplex(include_hyper_image=True)

    lens_galaxy = galaxy_prior.GalaxyPrior(sie_mass_profile=result_3.instance.sie_mass_profile,
                                           shear_mass_profile=result_3.instance.shear_mass_profile,
                                           sersic_light_profile=light_profiles.EllipticalSersic,
                                           exponential_light_profile=light_profiles.EllipticalExponential,
                                           hyper_galaxy=galaxy.HyperGalaxy)

    lens_galaxy.sersic_light_profile.centre = lens_galaxy.exponential_light_profile.centre
    lens_galaxy.sersic_light_profile.phi = lens_galaxy.exponential_light_profile.phi

    source_galaxy = galaxy_prior.GalaxyPrior(pixelization=pixelization.SquarePixelization)

    pixelization_result = result_3.priors.pixelization
    source_galaxy.pixelization.pixels = pixelization_result.pixels
    source_galaxy.pixelization.regularization_coefficients = pixelization_result.regularization_coefficients

    optimizer_4.variable.lens_galaxies = [lens_galaxy]
    optimizer_4.variable.source_galaxies = [source_galaxy]

    optimizer_4.fit(full_light_analysis)


default_config_path = '{}/../../config/'.format(os.path.dirname(os.path.realpath(__file__)))
default_data_path = '{}/../../weighted_data/'.format(os.path.dirname(os.path.realpath(__file__)))
# default_results_path = '{}/../../../results/'.format(os.path.dirname(os.path.realpath(__file__)))
default_results_path = '/home/jammy/results/'


class PipelinePaths(object):

    def __init__(self, data_name, config_path=default_config_path, data_path=default_data_path,
                 results_path=default_results_path):
        self.data_name = data_name
        self.config_path = config_path
        self.data_path = data_path + data_name + '/'
        self.results_path = results_path


def profiles_pipeline(paths, image, mask):
    """
     Fit an image with profiles, gradually adding extra profiles.

     Parameters
     ----------
     paths
     image: Image
         An image of a lens galaxy including metadata such as PSF, background noise and effective exposure time
     mask: Mask
         A mask describing which parts of the image should be excluded from the analysis

     Returns
     -------
     result: [Result]
         An array of results, one for each stage of the analysis
     """

    logger.info(
        """
        Pipeline Profiles:

        PURPOSE - Fit profiles to an image.

        PREPROCESSING:

        - N/A

        NOTES:

        Image: Observed image used throughout.
        Mask: Assume a large mask (e.g. 2.8") throughout - this value could be chosen in preprocessing.
        """
    )

    pipeline_name = 'Hyper_Lens'
    pipeline_path = paths.results_path + pipeline_name + '/'
    pipeline_data_path = pipeline_path + paths.data_name + '/'
    if not os.path.exists(pipeline_path): os.mkdir(pipeline_path)
    if not os.path.exists(pipeline_data_path): os.mkdir(pipeline_data_path)

    # Create an array in which to store results
    results = []

    # Create an analysis object for the image and mask
    analysis = an.Analysis(image, mask)

    logger.info(
        """
        1) Light: Elliptical Sersic x1
        """
    )

    analysis_path = pipeline_data_path + '1_lens_init/'

    # Create an optimizer
    optimizer_1 = non_linear.DownhillSimplex(path=analysis_path)

    # Define galaxy priors
    lens_galaxy_prior = galaxy_prior.GalaxyPrior(sersic_light_profile=light_profiles.EllipticalSersic)
    lens_galaxy_prior.sersic_light_profile.centre = model_mapper.Constant((0.0, 0.0))
    lens_galaxy_prior.sersic_light_profile.axis_ratio = model_mapper.Constant(0.8)
    lens_galaxy_prior.sersic_light_profile.phi = model_mapper.Constant(90.0)
    lens_galaxy_prior.sersic_light_profile.intensity = model_mapper.Constant(0.5)
    lens_galaxy_prior.sersic_light_profile.effective_radius = model_mapper.Constant(1.3)

    lens_galaxy_prior.redshift = model_mapper.Constant(1.0)

    # Add the galaxy priors to the optimizer
    optimizer_1.variable.lens_galaxies = [lens_galaxy_prior]

    # Analyse the system
    result_1 = optimizer_1.fit(analysis)

    # Add the result of the second analysis to the list
    results.append(result_1)

    logger.info(
        """
        Pipeline Profiles:

        PURPOSE - Fit profiles to an image.

        PREPROCESSING:

        - N/A

        NOTES:

        Image: Observed image used throughout.
        Mask: Assume a large mask (e.g. 2.8") throughout - this value could be chosen in preprocessing.
        """
    )

    # Create an array in which to store results
    results = []

    # TODO : These images should come from the results of the previous phase

    import numpy as np

    model_image = np.ones((mask.pixels_in_mask,))
    galaxy_images = [np.ones((mask.pixels_in_mask,))]

    # TODO : minimum value hsould be 0 if the galaxy was a profile, and 0.02 if it was a pixelization.

    minimum_values = [0.0]

    # Create an analysis object for the image and mask
    analysis = an.Analysis(image, mask, model_image=model_image, galaxy_images=galaxy_images,
                           minimum_values=minimum_values)

    logger.info(
        """
        2) Light: Elliptical Sersic x2 (The results of the first Sersic are fixed)
        """
    )

    analysis_path = pipeline_data_path + '1h_lens/'

    # Create an optimizer
    optimizer_1 = non_linear.DownhillSimplex(path=analysis_path)

    # Define galaxy priors
    lens_galaxy_prior = galaxy_prior.GalaxyPrior(
        sersic_light_profile=result_1.instance.lens_galaxies[0].sersic_light_profile,
        hyper_galaxy=galaxy.HyperGalaxy)

    # Add the galaxy priors to the optimizer
    optimizer_1.variable.lens_galaxies = [lens_galaxy_prior]

    # Analyse the system
    result_1 = optimizer_1.fit(analysis)

    # Add the result of the second analysis to the list
    results.append(result_1)


# def profiles_pipeline(paths, image, mask):
#     """
#      Fit an image with profiles, gradually adding extra profiles.
#
#      Parameters
#      ----------
#      image: Image
#          An image of a lens galaxy including metadata such as PSF, background noise and effective exposure time
#      mask: Mask
#          A mask describing which parts of the image should be excluded from the analysis
#
#      Returns
#      -------
#      result: [Result]
#          An array of results, one for each stage of the analysis
#      """
#
#     logger.info(
#         """
#         Pipeline Profiles:
#
#         PURPOSE - Fit profiles to an image.
#
#         PREPROCESSING:
#
#         - N/A
#
#         NOTES:
#
#         Image: Observed image used throughout.
#         Mask: Assume a large mask (e.g. 2.8") throughout - this value could be chosen in preprocessing.
#         """
#     )
#
#     pipeline_name = 'Profiles'
#     pipeline_path = paths.results_path+pipeline_name+'/'
#     pipeline_data_path = pipeline_path+paths.data_name+'/'
#     if not os.path.exists(pipeline_path): os.mkdir(pipeline_path)
#     if not os.path.exists(pipeline_data_path): os.mkdir(pipeline_data_path)
#
#     # Create an array in which to store results
#     results = []
#
#     # Create an analysis object for the image and mask
#     analysis = an.Analysis(image, mask)
#
#     logger.info(
#         """
#         1) Light: Elliptical Sersic x1
#         """
#     )
#
#     analysis_path = pipeline_data_path + '1_profile_1st/'
#
#     # Create an optimizer
#     optimizer_1 = non_linear.DownhillSimplex(path=analysis_path)
#
#     # Define galaxy priors
#     lens_galaxy_prior = galaxy_prior.GalaxyPrior(sersic_light_profile=light_profiles.EllipticalSersic)
#     lens_galaxy_prior.sersic_light_profile.centre = model_mapper.Constant((1.0, 1.0))
#     # TODO : Can we get rid of the x2 redshift?
#     lens_galaxy_prior.redshift.redshift = model_mapper.Constant(1.0)
#
#     # Add the galaxy priors to the optimizer
#     print(lens_galaxy_prior)
#     stop
#     optimizer_1.lens_galaxies = [lens_galaxy_prior]
#
#     # Analyse the system
#     result_1 = optimizer_1.fit(analysis)
#
#     # Add the result of the second analysis to the list
#     results.append(result_1)
#
#     logger.info(
#         """
#         Pipeline Profiles:
#
#         PURPOSE - Fit profiles to an image.
#
#         PREPROCESSING:
#
#         - N/A
#
#         NOTES:
#
#         Image: Observed image used throughout.
#         Mask: Assume a large mask (e.g. 2.8") throughout - this value could be chosen in preprocessing.
#         """
#     )
#
#     pipeline_name = 'Profiles'
#     pipeline_path = paths.results_path+pipeline_name+'/'
#     pipeline_data_path = pipeline_path+paths.data_name+'/'
#     if not os.path.exists(pipeline_path): os.mkdir(pipeline_path)
#     if not os.path.exists(pipeline_data_path): os.mkdir(pipeline_data_path)
#
#     # Create an array in which to store results
#     results = []
#
#     # Create an analysis object for the image and mask
#     analysis = an.Analysis(image, mask)
#
#     logger.info(
#         """
#         2) Light: Elliptical Sersic x2 (The results of the first Sersic are fixed)
#         """
#     )
#
#     analysis_path = pipeline_data_path + '1_profile_1st/'
#
#     # Create an optimizer
#     optimizer_1 = non_linear.DownhillSimplex(path=analysis_path)
#
#     # Define galaxy priors
#     lens_galaxy_prior = galaxy_prior.GalaxyPrior(sersic_light_profile=light_profiles.EllipticalSersic)
#     lens_galaxy_prior.sersic_light_profile.centre.centre_0 = model_mapper.GaussianPrior(1.0, 1.0)
#     lens_galaxy_prior.sersic_light_profile.centre.centre_1 = model_mapper.GaussianPrior(1.0, 1.0)
#
#     # Add the galaxy priors to the optimizer
#     optimizer_1.lens_galaxies = [lens_galaxy_prior]
#
#     # Analyse the system
#     result_1 = optimizer_1.fit(analysis)
#
#     # Add the result of the second analysis to the list
#     results.append(result_1)

"""

4H) Hyper-parameters: All included in model (most priors broad and uniform, but use previous phase regularization as well)
    Image: The lens light subtracted image from phase 1.
    Mask: Circle / Ellipse containing entire lens galaxy.
    Mass: SIE+Shear (Fixed to highest likelihood model from phase 2)
    Source: 'noisy' pixelization
    NLO: MN.
    """
