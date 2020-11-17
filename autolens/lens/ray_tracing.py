from abc import ABC
import pickle
import numpy as np
from os import path
from astropy import cosmology as cosmo
from autoarray.inversion import pixelizations as pix
from autoarray.inversion import inversions as inv
from autoarray.structures import grids
from autogalaxy import lensing
from autogalaxy.galaxy import galaxy as g
from autogalaxy.plane import plane as pl
from autogalaxy.util import cosmology_util
from autogalaxy.util import plane_util


class AbstractTracer(lensing.LensingObject, ABC):
    def __init__(self, planes, cosmology):
        """Ray-tracer for a lens system with any number of planes.

        The redshift of these planes are specified by the redshits of the galaxies; there is a unique plane redshift \
        for every unique galaxy redshift (galaxies with identical redshifts are put in the same plane).

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lens-geometry of the multi-plane system. All galaxies input to the tracer must therefore \
        have redshifts.

        This tracer has only one grid (see gridStack) which is used for ray-tracing.

        Parameters
        ----------
        galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_grid : grid_stacks.GridStack
            The image-plane grid which is traced. (includes the grid, sub-grid, blurring-grid, etc.).
        border : masks.GridBorder
            The border of the grid, which is used to relocate demagnified traced pixels to the \
            source-plane borders.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """
        self.planes = planes
        self.plane_redshifts = [plane.redshift for plane in planes]
        self.cosmology = cosmology

    @property
    def total_planes(self):
        return len(self.plane_redshifts)

    @property
    def image_plane(self):
        return self.planes[0]

    @property
    def source_plane(self):
        return self.planes[-1]

    @property
    def galaxies(self):
        return list([galaxy for plane in self.planes for galaxy in plane.galaxies])

    @property
    def all_planes_have_redshifts(self):
        return None not in self.plane_redshifts

    @property
    def has_light_profile(self):
        return any(list(map(lambda plane: plane.has_light_profile, self.planes)))

    @property
    def has_mass_profile(self):
        return any(list(map(lambda plane: plane.has_mass_profile, self.planes)))

    @property
    def has_pixelization(self):
        return any(list(map(lambda plane: plane.has_pixelization, self.planes)))

    @property
    def has_regularization(self):
        return any(list(map(lambda plane: plane.has_regularization, self.planes)))

    @property
    def has_hyper_galaxy(self):
        return any(list(map(lambda plane: plane.has_hyper_galaxy, self.planes)))

    @property
    def upper_plane_index_with_light_profile(self):
        return max(
            [
                plane_index if plane.has_light_profile else 0
                for (plane_index, plane) in enumerate(self.planes)
            ]
        )

    @property
    def planes_with_light_profile(self):
        return list(filter(lambda plane: plane.has_light_profile, self.planes))

    @property
    def planes_with_mass_profile(self):
        return list(filter(lambda plane: plane.has_mass_profile, self.planes))

    @property
    def light_profile_centres(self):
        """
        Returns the light profile centres of the tracer as a `GridCoordinates` object, which structures the centres
            in lists according to which plane they come from.

            Fo example, if the tracer has two planes, the first with one light profile and second with two light profiles
            this returns:

            [[(y0, x0)], [(y0, x0), (y1, x1)]]

            This is used for visualization, for example plotting the centres of all light profiles colored by their galaxy.

            The centres of light-sheets are filtered out, as their centres are not relevant to lensing calculations

        """
        return grids.GridCoordinates(
            [
                list(plane.light_profile_centres)
                for plane in self.planes
                if plane.has_light_profile
            ]
        )

    @property
    def mass_profiles(self):
        return [
            item
            for mass_profile in self.mass_profiles_of_planes
            for item in mass_profile
        ]

    @property
    def mass_profiles_of_planes(self):
        return [plane.mass_profiles for plane in self.planes if plane.has_mass_profile]

    @property
    def mass_profile_centres(self):
        """
        Returns the mass profile centres of the tracer as a `GridCoordinates` object, which structures the centres
            in lists according to which plane they come from.

            Fo example, if the tracer has two planes, the first with one mass profile and second with two mass profiles
            this returns:

            [[(y0, x0)], [(y0, x0), (y1, x1)]]

            This is used for visualization, for example plotting the centres of all mass profiles colored by their galaxy.

            The centres of mass-sheets are filtered out, as their centres are not relevant to lensing calculations

        """
        return grids.GridCoordinates(
            [
                list(plane.mass_profile_centres)
                for plane in self.planes
                if plane.has_mass_profile
            ]
        )

    @property
    def plane_indexes_with_pixelizations(self):
        plane_indexes_with_inversions = [
            plane_index if plane.has_pixelization else None
            for (plane_index, plane) in enumerate(self.planes)
        ]
        return [
            plane_index
            for plane_index in plane_indexes_with_inversions
            if plane_index is not None
        ]

    @property
    def pixelizations_of_planes(self):
        return [plane.pixelization for plane in self.planes]

    @property
    def regularizations_of_planes(self):
        return [plane.regularization for plane in self.planes]

    @property
    def hyper_galaxy_image_of_planes_with_pixelizations(self):
        return [
            plane.hyper_galaxy_image_of_galaxy_with_pixelization
            for plane in self.planes
        ]

    def plane_with_galaxy(self, galaxy):
        return [plane for plane in self.planes if galaxy in plane.galaxies][0]

    @classmethod
    def load(cls, file_path, filename="tracer"):
        with open(path.join(file_path, f"{filename}.pickle"), "rb") as f:
            return pickle.load(f)

    def save(self, file_path, filename="tracer"):
        """
        Save the tracer by serializing it with pickle.
        """
        with open(path.join(file_path, f"{filename}.pickle"), "wb") as f:
            pickle.dump(self, f)


class AbstractTracerLensing(AbstractTracer, ABC):
    @grids.grid_like_to_structure_list
    def traced_grids_of_planes_from_grid(self, grid, plane_index_limit=None):

        traced_grids = []
        traced_deflections = []

        for (plane_index, plane) in enumerate(self.planes):

            scaled_grid = grid.copy()

            if plane_index > 0:
                for previous_plane_index in range(plane_index):
                    scaling_factor = (
                        cosmology_util.scaling_factor_between_redshifts_from(
                            redshift_0=self.plane_redshifts[previous_plane_index],
                            redshift_1=plane.redshift,
                            redshift_final=self.plane_redshifts[-1],
                            cosmology=self.cosmology,
                        )
                    )

                    scaled_deflections = (
                        scaling_factor * traced_deflections[previous_plane_index]
                    )

                    # TODO : Setup as GridInterpolate

                    scaled_grid -= scaled_deflections

            traced_grids.append(scaled_grid)

            if plane_index_limit is not None:
                if plane_index == plane_index_limit:
                    return traced_grids

            traced_deflections.append(plane.deflections_from_grid(grid=scaled_grid))

        return traced_grids

    @grids.grid_like_to_structure
    def deflections_between_planes_from_grid(self, grid, plane_i=0, plane_j=-1):

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)

        return traced_grids_of_planes[plane_i] - traced_grids_of_planes[plane_j]

    @grids.grid_like_to_structure
    def image_from_grid(self, grid):
        return sum(self.images_of_planes_from_grid(grid=grid))

    @grids.grid_like_to_structure_list
    def images_of_planes_from_grid(self, grid):

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=grid, plane_index_limit=self.upper_plane_index_with_light_profile
        )

        images_of_planes = [
            self.planes[plane_index].image_from_grid(
                grid=traced_grids_of_planes[plane_index]
            )
            for plane_index in range(len(traced_grids_of_planes))
        ]

        if self.upper_plane_index_with_light_profile < self.total_planes - 1:
            for plane_index in range(
                self.upper_plane_index_with_light_profile, self.total_planes - 1
            ):
                images_of_planes.append(np.zeros(shape=images_of_planes[0].shape))

        return images_of_planes

    def padded_image_from_grid_and_psf_shape(self, grid, psf_shape_2d):

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=psf_shape_2d)

        return self.image_from_grid(grid=padded_grid)

    @grids.grid_like_to_structure
    def convergence_from_grid(self, grid):
        return sum([plane.convergence_from_grid(grid=grid) for plane in self.planes])

    @grids.grid_like_to_structure
    def potential_from_grid(self, grid):
        return sum([plane.potential_from_grid(grid=grid) for plane in self.planes])

    @grids.grid_like_to_structure
    def deflections_from_grid(self, grid):
        return self.deflections_between_planes_from_grid(grid=grid)

    @grids.grid_like_to_structure
    def deflections_of_planes_summed_from_grid(self, grid):
        return sum([plane.deflections_from_grid(grid=grid) for plane in self.planes])

    def grid_at_redshift_from_grid_and_redshift(self, grid, redshift):
        """For an input grid of (y,x) arc-second image-plane coordinates, ray-trace the coordinates to any redshift in \
        the strong lens configuration.

        This is performed using multi-plane ray-tracing and the existing redshifts and planes of the tracer. However, \
        any redshift can be input even if a plane does not exist there, including redshifts before the first plane \
        of the lens system.

        Parameters
        ----------
        grid : ndsrray or aa.Grid
            The image-plane grid which is traced to the redshift.
        redshift : float
            The redshift the image-plane grid is traced to.
        """

        if redshift <= self.plane_redshifts[0]:
            return grid.copy()

        plane_index_with_redshift = [
            plane_index
            for plane_index, plane in enumerate(self.planes)
            if plane.redshift == redshift
        ]

        if plane_index_with_redshift:
            return self.traced_grids_of_planes_from_grid(grid=grid)[
                plane_index_with_redshift[0]
            ]

        for plane_index, plane_redshift in enumerate(self.plane_redshifts):

            if redshift < plane_redshift:
                plane_index_insert = plane_index

        planes = self.planes
        planes.insert(plane_index_insert, pl.Plane(redshift=redshift, galaxies=[]))

        tracer = Tracer(planes=planes, cosmology=self.cosmology)

        return tracer.traced_grids_of_planes_from_grid(grid=grid)[plane_index_insert]

    @property
    def contribution_map(self):

        contribution_maps = self.contribution_maps_of_planes
        if None in contribution_maps:
            contribution_maps = [i for i in contribution_maps if i is not None]

        if contribution_maps:
            return sum(contribution_maps)
        else:
            return None

    @property
    def contribution_maps_of_planes(self):

        contribution_maps = []

        for plane in self.planes:

            if plane.contribution_map is not None:

                contribution_maps.append(plane.contribution_map)

            else:

                contribution_maps.append(None)

        return contribution_maps


class AbstractTracerData(AbstractTracerLensing, ABC):
    def blurred_image_from_grid_and_psf(self, grid, psf, blurring_grid):
        """Extract the 1D image and 1D blurring image of every plane and blur each with the \
        PSF using a psf (see imaging.convolution).

        These are summed to give the tracer's overall blurred image in 1D.

        Parameters
        ----------
        psf : hyper_galaxies.imaging.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        if not self.has_light_profile:
            return np.zeros(shape=grid.shape_1d)

        image = self.image_from_grid(grid=grid)

        blurring_image = self.image_from_grid(grid=blurring_grid)

        return psf.convolved_array_from_array_2d_and_mask(
            array_2d=image.in_2d_binned + blurring_image.in_2d_binned, mask=grid.mask
        )

    def blurred_images_of_planes_from_grid_and_psf(self, grid, psf, blurring_grid):
        """Extract the 1D image and 1D blurring image of every plane and blur each with the \
        PSF using a psf (see imaging.convolution).

        The blurred image of every plane is returned in 1D.

        Parameters
        ----------
        psf : hyper_galaxies.imaging.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)
        traced_blurring_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=blurring_grid
        )
        return [
            plane.blurred_image_from_grid_and_psf(
                grid=traced_grids_of_planes[plane_index],
                psf=psf,
                blurring_grid=traced_blurring_grids_of_planes[plane_index],
            )
            for (plane_index, plane) in enumerate(self.planes)
        ]

    def blurred_image_from_grid_and_convolver(self, grid, convolver, blurring_grid):
        """Extract the 1D image and 1D blurring image of every plane and blur each with the \
        PSF using a convolver (see imaging.convolution).

        These are summed to give the tracer's overall blurred image in 1D.

        Parameters
        ----------
        convolver : hyper_galaxies.imaging.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        if not self.has_light_profile:
            return np.zeros(shape=grid.shape_1d)

        image = self.image_from_grid(grid=grid)

        blurring_image = self.image_from_grid(grid=blurring_grid)

        return convolver.convolved_image_from_image_and_blurring_image(
            image=image, blurring_image=blurring_image
        )

    def blurred_images_of_planes_from_grid_and_convolver(
        self, grid, convolver, blurring_grid
    ):
        """Extract the 1D image and 1D blurring image of every plane and blur each with the \
        PSF using a convolver (see imaging.convolution).

        The blurred image of every plane is returned in 1D.

        Parameters
        ----------
        convolver : hyper_galaxies.imaging.convolution.ConvolverImage
            Class which performs the PSF convolution of a masked image in 1D.
        """

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)
        traced_blurring_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=blurring_grid
        )

        return [
            plane.blurred_image_from_grid_and_convolver(
                grid=traced_grids_of_planes[plane_index],
                convolver=convolver,
                blurring_grid=traced_blurring_grids_of_planes[plane_index],
            )
            for (plane_index, plane) in enumerate(self.planes)
        ]

    def unmasked_blurred_image_from_grid_and_psf(self, grid, psf):

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=psf.shape_2d)

        padded_image = self.image_from_grid(grid=padded_grid)

        return padded_grid.mask.unmasked_blurred_array_from_padded_array_psf_and_image_shape(
            padded_array=padded_image, psf=psf, image_shape=grid.mask.shape
        )

    def unmasked_blurred_image_of_planes_from_grid_and_psf(self, grid, psf):

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=psf.shape_2d)

        traced_padded_grids = self.traced_grids_of_planes_from_grid(grid=padded_grid)

        unmasked_blurred_images_of_planes = []

        for plane, traced_padded_grid in zip(self.planes, traced_padded_grids):
            padded_image_1d = plane.image_from_grid(grid=traced_padded_grid)

            unmasked_blurred_array_2d = padded_grid.mask.unmasked_blurred_array_from_padded_array_psf_and_image_shape(
                padded_array=padded_image_1d, psf=psf, image_shape=grid.mask.shape
            )

            unmasked_blurred_images_of_planes.append(unmasked_blurred_array_2d)

        return unmasked_blurred_images_of_planes

    def unmasked_blurred_image_of_planes_and_galaxies_from_grid_and_psf(
        self, grid, psf
    ):

        unmasked_blurred_images_of_planes_and_galaxies = []

        padded_grid = grid.padded_grid_from_kernel_shape(kernel_shape_2d=psf.shape_2d)

        traced_padded_grids = self.traced_grids_of_planes_from_grid(grid=padded_grid)

        for plane, traced_padded_grid in zip(self.planes, traced_padded_grids):
            padded_image_1d_of_galaxies = plane.images_of_galaxies_from_grid(
                grid=traced_padded_grid
            )

            unmasked_blurred_array_2d_of_galaxies = list(
                map(
                    lambda padded_image_1d_of_galaxy: padded_grid.mask.unmasked_blurred_array_from_padded_array_psf_and_image_shape(
                        padded_array=padded_image_1d_of_galaxy,
                        psf=psf,
                        image_shape=grid.mask.shape,
                    ),
                    padded_image_1d_of_galaxies,
                )
            )

            unmasked_blurred_images_of_planes_and_galaxies.append(
                unmasked_blurred_array_2d_of_galaxies
            )

        return unmasked_blurred_images_of_planes_and_galaxies

    def profile_visibilities_from_grid_and_transformer(self, grid, transformer):

        if not self.has_light_profile:
            return np.zeros(shape=transformer.uv_wavelengths.shape)

        image = self.image_from_grid(grid=grid)

        return transformer.visibilities_from_image(image=image)

    def profile_visibilities_of_planes_from_grid_and_transformer(
        self, grid, transformer
    ):

        images_of_planes = self.images_of_planes_from_grid(grid=grid)
        return [
            transformer.visibilities_from_image(image=image)
            for image in images_of_planes
        ]

    def sparse_image_plane_grids_of_planes_from_grid(
        self, grid, pixelization_setting=pix.SettingsPixelization()
    ):

        sparse_image_plane_grids_of_planes = []

        for plane in self.planes:
            sparse_image_plane_grid = plane.sparse_image_plane_grid_from_grid(
                grid=grid, settings_pixelization=pixelization_setting
            )
            sparse_image_plane_grids_of_planes.append(sparse_image_plane_grid)

        return sparse_image_plane_grids_of_planes

    def traced_sparse_grids_of_planes_from_grid(
        self, grid, settings_pixelization=pix.SettingsPixelization()
    ):

        if (
            settings_pixelization.preload_sparse_grids_of_planes is None
            or settings_pixelization.is_stochastic
        ):

            sparse_image_plane_grids_of_planes = (
                self.sparse_image_plane_grids_of_planes_from_grid(
                    grid=grid, pixelization_setting=settings_pixelization
                )
            )

        else:

            sparse_image_plane_grids_of_planes = (
                settings_pixelization.preload_sparse_grids_of_planes
            )

        traced_sparse_grids_of_planes = []

        for (plane_index, plane) in enumerate(self.planes):

            if sparse_image_plane_grids_of_planes[plane_index] is None:
                traced_sparse_grids_of_planes.append(None)
            else:
                traced_sparse_grids = self.traced_grids_of_planes_from_grid(
                    grid=sparse_image_plane_grids_of_planes[plane_index]
                )
                traced_sparse_grids_of_planes.append(traced_sparse_grids[plane_index])

        return traced_sparse_grids_of_planes

    def mappers_of_planes_from_grid(
        self, grid, settings_pixelization=pix.SettingsPixelization()
    ):

        mappers_of_planes = []

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)

        traced_sparse_grids_of_planes = self.traced_sparse_grids_of_planes_from_grid(
            grid=grid, settings_pixelization=settings_pixelization
        )

        for (plane_index, plane) in enumerate(self.planes):

            if not plane.has_pixelization:
                mappers_of_planes.append(None)
            else:
                mapper = plane.mapper_from_grid_and_sparse_grid(
                    grid=traced_grids_of_planes[plane_index],
                    sparse_grid=traced_sparse_grids_of_planes[plane_index],
                    settings_pixelization=settings_pixelization,
                )
                mappers_of_planes.append(mapper)

        return mappers_of_planes

    def inversion_imaging_from_grid_and_data(
        self,
        grid,
        image,
        noise_map,
        convolver,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
    ):

        mappers_of_planes = self.mappers_of_planes_from_grid(
            grid=grid, settings_pixelization=settings_pixelization
        )

        return inv.InversionImagingMatrix.from_data_mapper_and_regularization(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            mapper=mappers_of_planes[-1],
            regularization=self.regularizations_of_planes[-1],
            settings=settings_inversion,
        )

    def inversion_interferometer_from_grid_and_data(
        self,
        grid,
        visibilities,
        noise_map,
        transformer,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
    ):
        mappers_of_planes = self.mappers_of_planes_from_grid(
            grid=grid, settings_pixelization=settings_pixelization
        )

        return inv.AbstractInversionInterferometer.from_data_mapper_and_regularization(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            mapper=mappers_of_planes[-1],
            regularization=self.regularizations_of_planes[-1],
            settings=settings_inversion,
        )

    def hyper_noise_map_from_noise_map(self, noise_map):
        hyper_noise_maps = self.hyper_noise_maps_of_planes_from_noise_map(
            noise_map=noise_map
        )
        return sum(hyper_noise_maps)

    def hyper_noise_maps_of_planes_from_noise_map(self, noise_map):
        return [
            plane.hyper_noise_map_from_noise_map(noise_map=noise_map)
            for plane in self.planes
        ]

    def galaxy_image_dict_from_grid(self, grid) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_image_dict = dict()

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)

        for (plane_index, plane) in enumerate(self.planes):
            images_of_galaxies = plane.images_of_galaxies_from_grid(
                grid=traced_grids_of_planes[plane_index]
            )
            for (galaxy_index, galaxy) in enumerate(plane.galaxies):
                galaxy_image_dict[galaxy] = images_of_galaxies[galaxy_index]

        return galaxy_image_dict

    def galaxy_blurred_image_dict_from_grid_and_convolver(
        self, grid, convolver, blurring_grid
    ) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_blurred_image_dict = dict()

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)

        traced_blurring_grids_of_planes = self.traced_grids_of_planes_from_grid(
            grid=blurring_grid
        )

        for (plane_index, plane) in enumerate(self.planes):
            blurred_images_of_galaxies = (
                plane.blurred_images_of_galaxies_from_grid_and_convolver(
                    grid=traced_grids_of_planes[plane_index],
                    convolver=convolver,
                    blurring_grid=traced_blurring_grids_of_planes[plane_index],
                )
            )
            for (galaxy_index, galaxy) in enumerate(plane.galaxies):
                galaxy_blurred_image_dict[galaxy] = blurred_images_of_galaxies[
                    galaxy_index
                ]

        return galaxy_blurred_image_dict

    def galaxy_profile_visibilities_dict_from_grid_and_transformer(
        self, grid, transformer
    ) -> {g.Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_profile_visibilities_image_dict = dict()

        traced_grids_of_planes = self.traced_grids_of_planes_from_grid(grid=grid)

        for (plane_index, plane) in enumerate(self.planes):
            profile_visibilities_of_galaxies = (
                plane.profile_visibilities_of_galaxies_from_grid_and_transformer(
                    grid=traced_grids_of_planes[plane_index], transformer=transformer
                )
            )
            for (galaxy_index, galaxy) in enumerate(plane.galaxies):
                galaxy_profile_visibilities_image_dict[
                    galaxy
                ] = profile_visibilities_of_galaxies[galaxy_index]

        return galaxy_profile_visibilities_image_dict


class Tracer(AbstractTracerData):
    @classmethod
    def from_galaxies(cls, galaxies, cosmology=cosmo.Planck15):

        plane_redshifts = plane_util.ordered_plane_redshifts_from(galaxies=galaxies)

        galaxies_in_planes = plane_util.galaxies_in_redshift_ordered_planes_from(
            galaxies=galaxies, plane_redshifts=plane_redshifts
        )

        planes = []

        for plane_index in range(0, len(plane_redshifts)):
            planes.append(pl.Plane(galaxies=galaxies_in_planes[plane_index]))

        return Tracer(planes=planes, cosmology=cosmology)

    @classmethod
    def sliced_tracer_from_lens_line_of_sight_and_source_galaxies(
        cls,
        lens_galaxies,
        line_of_sight_galaxies,
        source_galaxies,
        planes_between_lenses,
        cosmology=cosmo.Planck15,
    ):

        """Ray-tracer for a lens system with any number of planes.

        The redshift of these planes are specified by the input parameters *lens_redshifts* and \
         *slices_between_main_planes*. Every galaxy is placed in its closest plane in redshift-space.

        To perform multi-plane ray-tracing, a cosmology must be supplied so that deflection-angles can be rescaled \
        according to the lens-geometry of the multi-plane system. All galaxies input to the tracer must therefore \
        have redshifts.

        This tracer has only one grid (see gridStack) which is used for ray-tracing.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of galaxies in the ray-tracing calculation.
        image_plane_grid : grid_stacks.GridStack
            The image-plane grid which is traced. (includes the grid, sub-grid, blurring-grid, etc.).
        planes_between_lenses : [int]
            The number of slices between each main plane. The first entry in this list determines the number of slices \
            between Earth (redshift 0.0) and main plane 0, the next between main planes 0 and 1, etc.
        border : masks.GridBorder
            The border of the grid, which is used to relocate demagnified traced pixels to the \
            source-plane borders.
        cosmology : astropy.cosmology
            The cosmology of the ray-tracing calculation.
        """

        lens_redshifts = plane_util.ordered_plane_redshifts_from(galaxies=lens_galaxies)

        plane_redshifts = plane_util.ordered_plane_redshifts_with_slicing_from(
            lens_redshifts=lens_redshifts,
            planes_between_lenses=planes_between_lenses,
            source_plane_redshift=source_galaxies[0].redshift,
        )

        galaxies_in_planes = plane_util.galaxies_in_redshift_ordered_planes_from(
            galaxies=lens_galaxies + line_of_sight_galaxies,
            plane_redshifts=plane_redshifts,
        )

        plane_redshifts.append(source_galaxies[0].redshift)
        galaxies_in_planes.append(source_galaxies)

        planes = []

        for plane_index in range(0, len(plane_redshifts)):
            planes.append(
                pl.Plane(
                    redshift=plane_redshifts[plane_index],
                    galaxies=galaxies_in_planes[plane_index],
                )
            )

        return Tracer(planes=planes, cosmology=cosmology)
