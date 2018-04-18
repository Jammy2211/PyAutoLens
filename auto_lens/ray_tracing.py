from auto_lens.profiles import geometry_profiles

import numpy as np


class RayTrace(object):
    """The ray-tracing calculations, defined by the image_grid and source planes of every galaxy in this lensing system.
    These are computed in order of ascending galaxy redshift.

    The ray-tracing calculations between every plane is determined using the angular diameter distances between each \
    set of galaxies.

    This is used to perform all ray-tracing calculations and for converting dimensionless measurements (e.g. \
    arc-seconds, mass) to physical units.

    Parameters
    ----------
    galaxies : [Galaxy]
        The list of galaxies that form the lensing planes.
    cosmological_model : astropy.cosmology.FLRW
        The assumed cosmology for this ray-tracing calculation.
    """
    pass


class TraceImageAndSource(object):

    def __init__(self, lens_galaxies, source_galaxies, image_plane_grids):
        """The ray-tracing calculations, defined by a lensing system with just one image_grid-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need \
        known redshift measurements. For computational efficiency, it is recommend this ray-tracing class is used for \
        lens modeling, provided cosmological information is not necessary.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the image_grid-plane.
        source_galaxies : [Galaxy]
            The list of source galaxies in the source-plane.
        image_plane_grids : RayTracingGrids
            The image_grid of the ray-tracing calculation, (includes the image.grid, sub_grid, \
            blurring_grid region etc.), which begins in the image_grid-plane.
        """
        self.image_plane = ImagePlane(lens_galaxies, image_plane_grids)

        source_plane_coordinates = self.image_plane.trace_to_next_plane()

        self.source_plane = SourcePlane(source_galaxies, source_plane_coordinates)


class Plane(object):
    """Represents a plane of galaxies and grids.

    Parameters
    ----------
    galaxies : [Galaxy]
        The galaxies in the plane.
    plane_coordinates : PlaneCoordinates
        The x and y image_grid in the plane. Includes all image_grid e.g. the image_grid, sub_grid-grid, sparse_grid-grid, etc.
    """
    def __init__(self, galaxies, grids):

        self.galaxies = galaxies
        self.grids = grids


class LensPlane(Plane):

    def __init__(self, galaxies, grids):
        """Represents a lens-plane, a set of galaxies and grids at an intermediate redshift in the lens \
        ray-tracing calculation.

        A lens-plane is not the final ray-tracing plane and its grids will be traced another, higher \
        redshift, plane. Thus, the deflection angles due to the plane's galaxies are calculated.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the image_grid-plane.
        grids : PlaneCoordinates
            The x and y image_grid in the plane. Includes all image_grid e.g. the image_grid, sub_grid-grid, sparse_grid-grid, etc.
        """

        super(LensPlane, self).__init__(galaxies, grids)

        self.deflection_angles = self.grids.deflection_grids_from_galaxies(galaxies)

    def trace_to_next_plane(self):
        """Trace the image_grid pixel image_grid to the next plane, the source-plane.

        NOTE : This does not work for multiplane lensing, which requires one to use the previous plane's deflection \
        angles to perform the tracing. I guess we'll ultimately call this class 'LensPlanes' and have it as a list."""
        return self.grids.trace_to_next_grid(self.deflection_angles)

class ImagePlane(LensPlane):

    def __init__(self, galaxies, grids):
        """Represents an image_grid-plane, a set of galaxies and grids at the lowest redshift in the lens \
        ray-tracing calculation.

        The image_grid-plane is, by definition, a lens-plane, thus the deflection angles at each grids are computed.

        The image_grid-plane coodinates are defined on the observed image_grid's uniform regular grid. Calculating its light \
        profiles therefore exploits this uniformity to perform more efficient and precise calculations.

        The light profiles of galaxies at higher redshifts (and therefore in different lens-planes) can be assigned to \
        the ImagePlane instead. This occurs when:

        1) The efficiency and precision offered by computing the light profile on a uniform grid is preferred and \
        won't lead noticeable inaccuracy. For example, computing the light profile of main lens galaxy, ignoring \
        minor lensing effects due to a low mass foreground substructure.

        2) If evaluating the light profile in its lens-plane is inaccurate. For example, when modeling the \
        point-source images of a lensed quasar, effects like micro-lensing means lens-plane modeling will be inaccurate.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the image_grid-plane.
        grids : PlaneCoordinates
            The x and y image_grid in the plane. Includes all image_grid e.g. the image_grid, sub_grid-grid, sparse_grid-grid, etc.
        """

        super(ImagePlane, self).__init__(galaxies, grids)


class SourcePlane(Plane):

    def __init__(self, galaxies, grids):
        """Represents a source-plane, a set of galaxies and grids at the highest redshift in the lens \
        ray-tracing calculation.

        A source-plane is the final ray-tracing plane, thus the deflection angles due to the plane's galaxies are \
        not calculated.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the source-plane.
        grids : PlaneCoordinates
            The x and y image_grid in the plane. Includes all image_grid e.g. the image_grid, sub_grid-grid, sparse_grid-grid, etc.
        """
        super(SourcePlane, self).__init__(galaxies, grids)