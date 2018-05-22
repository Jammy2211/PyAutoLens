class TraceImageAndSource(object):

    def __init__(self, lens_galaxies, source_galaxies, image_plane_grids):
        """The ray-tracing calculations, defined by a lensing system with just one image-plane and source-plane.

        This has no associated cosmology, thus all calculations are performed in arc seconds and galaxies do not need
        known redshift measurements. For computational efficiency, it is recommend this ray-tracing class is used for
        lens modeling, provided cosmological information is not necessary.

        Parameters
        ----------
        lens_galaxies : [Galaxy]
            The list of lens galaxies in the image-plane.
        source_galaxies : [Galaxy]
            The list of source galaxies in the source-plane.
        image_plane_grids : GridCoordsCollection
            The image-plane grids of coordinates where ray-tracing calculation are performed, (this includes the
            image.grid_coords, sub_grid, blurring.grid_coords etc.).
        """
        self.image_plane = ImagePlane(lens_galaxies, image_plane_grids)

        source_plane_grids = self.image_plane.trace_to_next_plane()

        self.source_plane = SourcePlane(source_galaxies, source_plane_grids)

    def generate_image_of_galaxies(self):
        """Generate the image of the galaxies over the entire ray trace."""
        return self.image_plane.generate_image_of_galaxies() + self.source_plane.generate_image_of_galaxies()


class Plane(object):

    def __init__(self, galaxies, grids):
        """
        Represents a plane of galaxies and grids.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the plane.
        grids : grids.GridCoordsCollection
            The grids of (x,y) coordinates in the plane, including the image grid_coords, sub-grid_coords, blurring grid_coords, etc.
        """

        self.galaxies = galaxies
        self.grids = grids

    def generate_image_of_galaxies(self):
        """Generate the image of the galaxies in this plane."""
        return self.grids.image.intensities_via_grid(self.galaxies)


class LensPlane(Plane):

    def __init__(self, galaxies, grids):
        """Represents a lens-plane, a set of galaxies and grids at an intermediate redshift in the ray-tracing
        calculation.

        A lens-plane is not the final ray-tracing plane and its grids will be traced too another higher 
        redshift plane. Thus, the deflection angles due to the plane's galaxies are calculated.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the image_grid-plane.
        grids : grids.GridCoordsCollection
            The grids of (x,y) coordinates in the plane, including the image grid_coords, sub-grid_coords, blurring
            grid_coords, etc.
        """

        super(LensPlane, self).__init__(galaxies, grids)

        self.deflections = self.deflections_on_all_grids()

    def deflections_on_all_grids(self):
        """Compute the deflection angles on the grids"""
        return self.grids.deflection_grids_for_galaxies(self.galaxies)

    def trace_to_next_plane(self):
        """Trace the grids to the next plane.

        NOTE : This does not work for multi-plane lensing, which requires one to use the previous plane's deflection
        angles to perform the tracing. I guess we'll ultimately call this class 'LensPlanes' and have it as a list.
        """
        return self.grids.traced_grids_for_deflections(self.deflections)


# TODO: Do we need separate image and source planes? Could everything implicitly be a Plane?
class ImagePlane(LensPlane):

    def __init__(self, galaxies, grids):
        """Represents an image-plane, a set of galaxies and grids at the lowest redshift in the lens ray-tracing 
        calculation.

        The image-plane is, by definition, a lens-plane, thus the deflection angles for each grid_coords are computed.

        The image-plane coordinates are defined on the observed image's uniform regular grid_coords. Calculating its
        model images from its light profiles exploits this uniformity to perform more efficient and precise calculations
        via an iterative sub-griding approach.

        The light profiles of galaxies at higher redshifts (and therefore in different lens-planes) can be assigned to 
        the ImagePlane. This occurs when:

        1) The efficiency and precision offered by computing the light profile on a uniform grid_coords is preferred and 
        won't lead noticeable inaccuracy. For example, computing the light profile of the main lens galaxy, ignoring 
        minor lensing effects due to a low mass foreground substructure.

        2) When evaluating the light profile in its lens-plane is inaccurate. For example, when modeling the 
        point-source images of a lensed quasar, effects like micro-lensing mean lens-plane modeling will be inaccurate.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the image_grid-plane.
        grids : grids.GridCoordsCollection
            The grids of (x,y) coordinates in the plane, including the image grid_coords, sub-grid_coords, blurring
            grid_coords, etc.
        """

        super(ImagePlane, self).__init__(galaxies, grids)


class SourcePlane(Plane):

    def __init__(self, galaxies, grids):
        """Represents a source-plane, a set of galaxies and grids at the highest redshift in the ray-tracing 
        calculation.

        A source-plane is the final ray-tracing plane, thus the deflection angles due to the plane's galaxies are 
        not calculated.

        Parameters
        ----------
        galaxies : [Galaxy]
            The galaxies in the source-plane.
        grids : grids.GridCoordsCollection
            The grids of (x,y) coordinates in the plane, including the image grid_coords, sub-grid_coords, blurring
            grid_coords, etc.
        """
        super(SourcePlane, self).__init__(galaxies, grids)
