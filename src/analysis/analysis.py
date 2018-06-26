from src.analysis import fitting
from src.imaging import grids
from src.analysis import ray_tracing
from src.pixelization import frame_convolution
from src import exc
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

empty_array = []


class Analysis(object):
    def __init__(self, image, mask, grid_size_sub=4, cluster_grid_size=3):
        """
        An analysis object. Once set up with an image and mask it takes a set of objects describing a model and
        determines how well they fit the image.

        Parameters
        ----------
        image: Image
            An image of a lens with associated metadata
        mask: Mask
            A mask describing the region of the image to be modelled
        grid_size_sub: int
            The size of the sub-pixel grid for which values should be calculated
        cluster_grid_size: int:
            The sparsity of pixels to be used in clustering. Specifies the number of pixels to jump, meaning a higher
            number gives a lower density.
        """
        self.image = image
        self.mask = mask
        
        self.data_collection = grids.DataCollection.from_mask(mask, image, image.background_noise,
                                                              image.effective_exposure_time)
        self.coords_collection = grids.CoordsCollection.from_mask(mask, grid_size_sub=grid_size_sub,
                                                                  blurring_shape=image.psf.shape)
        self.mapper_collection = grids.MapperCollection.from_mask(mask, cluster_grid_size)

        self.mapper_cluster = grids.MapperCluster.from_mask(mask, cluster_grid_size)

        self.kernel_convolver = frame_convolution.FrameMaker(mask=mask).convolver_for_kernel(image.psf)

        logger.info("Analysis created for image "
                    "with shape {}, grid_sub_size {} and cluster_grid_size {}".format(image.shape,
                                                                                      grid_size_sub,
                                                                                      cluster_grid_size))

    def run(self, lens_galaxies=empty_array, source_galaxies=empty_array, hyper_image=None):
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
            logger.debug("Fitting for profiles")
            likelihood = fitting.fit_data_with_profiles(self.data_collection, self.kernel_convolver, tracer,
                                                        hyper_image)

        if likelihood is None:
            raise exc.PriorException("No galaxy has a profile or pixelization")

        return Analysis.Result(likelihood, lens_galaxies, source_galaxies, hyper_image)

    class Result(object):
        def __init__(self, likelihood, lens_galaxies, source_galaxies, hyper_image):
            self.likelihood = likelihood
            self.lens_galaxies = lens_galaxies
            self.source_galaxies = source_galaxies
            self.instrumentation = hyper_image

        def __str__(self):
            return "Analysis Result:\n{}".format(
                "\n".join(["{}: {}".format(key, value) for key, value in self.__dict__.items()]))
