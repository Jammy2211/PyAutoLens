from autolens.analysis import fitting
from autolens.analysis import ray_tracing
from imaging import convolution
from autolens import exc
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

empty_array = []


class Analysis(object):

    def __init__(self, image, mask, sub_grid_size=4, cluster_grid_size=3, model_image=None, galaxy_images=None,
                 minimum_values=None):
        """
        An analysis object. Once set up with an image and mask it takes a set of objects describing a model and
        determines how well they fit the image.

        Parameters
        ----------
        image: Image
            An image of a lens with associated metadata
        mask: Mask
            A mask describing the region of the image to be modelled
        sub_grid_size: int
            The sub_grid_size of the sub-pixel grid for which values should be calculated
        cluster_grid_size: int:
            The sparsity of pixels to be used in clustering. Specifies the number of pixels to jump, meaning a higher
            number gives a lower density.
        """
        self.image = image
        self.mask = mask

        self.data_collection = mask.data_collection_from_image_noise_and_exposure_time(image, image.background_noise,
                                                                                       image.effective_exposure_time)
        self.coords_collection = mask.coordinates_collection_for_subgrid_size_and_blurring_shape(
            sub_grid_size=sub_grid_size, blurring_shape=image.psf.shape)

        self.mapper_cluster = mask.sparse_grid_mapper_with_grid_size(cluster_grid_size)
        self.mapping = mask.grid_mapping_with_sub_grid_size(sub_grid_size)

        self.kernel_convolver = convolution.Convolver(mask=mask).convolver_for_kernel(image.psf)

        self.model_image = model_image
        self.galaxy_images = galaxy_images
        self.minimum_values = minimum_values

        logger.info("Analysis created for image "
                    "with shape {}, grid_sub_size {} and cluster_grid_size {}".format(image.shape,
                                                                                      sub_grid_size,
                                                                                      cluster_grid_size))

    def fit(self, lens_galaxies=empty_array, source_galaxies=empty_array, hyper_image=None):
        """
        Runs the analysis. Determines how well the supplied model fits the image.

        Parameters
        ----------
        lens_galaxies: [Galaxy]
            A collection of galaxies that form the lens
        source_galaxies: [Galaxy]
            A collection of galaxies that are being lensed
        hyper_image: HyperImage
            A class describing instrumental effects

        Returns
        -------
        result: Result
            An object comprising the final model instances generated and a corresponding likelihood
        """
        logger.debug(
            "\nRunning analysis for... \n\nLens Galaxies:\n{}\n\nSource Galaxies:\n{}\n\n".format(
                "\n\n".join(map(str, lens_galaxies)),
                "\n\n".join(
                    map(str, source_galaxies))))
        if hyper_image is not None:
            logger.debug("Hyper Image:\n{}".format(hyper_image))

        tracer = ray_tracing.Tracer(lens_galaxies, source_galaxies, self.coords_collection)

        galaxies = lens_galaxies + source_galaxies

        is_profile = True in map(lambda galaxy: galaxy.has_profile, galaxies)
        is_pixelization = True in map(lambda galaxy: galaxy.has_pixelization, galaxies)
        is_hyper_galaxy = True in map(lambda galaxy: galaxy.has_hyper_galaxy, galaxies)

        likelihood = None

        if is_pixelization:
            pixelized_galaxies = list(filter(lambda galaxy: galaxy.has_pixelization, galaxies))
            if len(pixelized_galaxies) > 0:
                raise exc.PriorException("Only one galaxy should have a pixelization")
            pixelized_galaxy = pixelized_galaxies[0]
            if pixelized_galaxy.has_profile:
                raise exc.PriorException("Galaxies should have either a pixelization or a profile")
            pixelization = pixelized_galaxy.pixelization
            if is_profile:
                logger.debug("Fitting for pixelization and profiles")
                likelihood = fitting.fit_data_with_pixelization_and_profiles(self.data_collection, pixelization,
                                                                             self.kernel_convolver, tracer,
                                                                             self.mapper_cluster, hyper_image)
            else:
                logger.debug("Fitting for pixelization")
                likelihood = fitting.fit_data_with_pixelization(self.data_collection, pixelization,
                                                                self.kernel_convolver, tracer, self.mapper_cluster,
                                                                hyper_image)
        elif is_profile:
            if not is_hyper_galaxy:
                logger.debug("Fitting for profiles (no hyper galaxy)")
                likelihood = fitting.fit_data_with_profiles(self.data_collection, self.kernel_convolver, tracer,
                                                            self.mapping, hyper_image)
            elif is_hyper_galaxy:
                logger.debug("Fitting for profiles (includes hyper galaxy)")

                # TODO : Extract list of hyper galaixes elegent (not all galaxies are necessary hyper gals)

                hyper_galaxies = [galaxies[0].hyper_galaxy]

                likelihood = fitting.fit_data_with_profiles_hyper_galaxies(self.data_collection, self.kernel_convolver,
                                                                           tracer, self.model_image, self.galaxy_images,
                                                                           self.minimum_values, hyper_galaxies)

        if likelihood is None:
            raise exc.PriorException("No galaxy has a profile or pixelization")

        return likelihood
