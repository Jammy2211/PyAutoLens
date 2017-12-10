import itertools
import math

class SourcePlane(object):
    """Represents the source-plane, including the traced image coordinates, the pixelization and regularization
    matrix"""

    def __init__(self, coordinates, sub_coordinates=None, centre=(0, 0)):
        """

        Parameters
        ----------
        coordinates : list(float, float)
            The x and y coordinates of each traced image pixel
        sub_coordinates : list(float, float)
            The x and y coordinates of each traced image sub-pixel
        centre : (float, float)
            The centre of the source-plane. For a single mass profile, this will be (0,0), but may be more complicated
            for multiple profiles.
        """
        self.coordinates = coordinates

        if sub_coordinates != None:
            self.sub_coordinates = sub_coordinates

        self.centre = centre

    @property
    def x_cen(self):
        return self.centre[0]

    @property
    def y_cen(self):
        return self.centre[1]

    def coordinates_to_centre(self, coordinates):
        """
        Converts source-plane coordinates to centre of source-plane

        Parameters
        ----------
        coordinates : (float, float)
            The x and y coordinates of the source-plane coordinate

        Returns
        ----------
        The coordinates at the source-plane center
        """
        return coordinates[0] - self.x_cen, coordinates[1] - self.y_cen

    def coordinates_to_radius(self, coordinates):
        """
        Convert the coordinates to a radius

        Parameters
        ----------
        coordinates : (float, float)
            The image coordinates (x, y)

        Returns
        -------
        The radius at those coordinates
        """
        shifted_coordinates = self.coordinates_to_centre(coordinates)
        return math.sqrt(shifted_coordinates[0] ** 2 + shifted_coordinates[1] ** 2)

    def compute_edge_function(self, edge_mask):
        """ Fit a function, r(theta), to the source-plane coordinates which are defined to be at its edge.

        Parameters
        ----------
        edge_mask : list(bool)
            A list of length coordinates indicating whether each coordinate is at the edge of the source-plane.
            *True* - This pixel is at the source-plane edge.
            *False* - Thiss pixel is not at the source-plane edge

        """

        coordinates_edge = list(itertools.compress(self.coordinates, edge_mask))
   #     radius_edge = list(map(lambda x, y : math.sqrt self.coordinates

