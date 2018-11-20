import numpy as np
from profiling import profiling_data
from profiling import tools

from profiles import light_profiles


class EllipticalSersic(light_profiles.EllipticalLightProfile):

    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=0.1, effective_radius=0.6,
                 sersic_index=4.0, settings=light_profiles.LightProfileSettings()):
        """

        Parameters
        ----------
        centre: (float, float)
            The image_grid of the origin of the profiles
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a)
        phi : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        intensity : float
            Overall intensity normalisation in the light profiles (electrons per second)
        effective_radius : float
            The circular radius containing half the light of this model_mapper
        sersic_index : Int
            The concentration of the light profiles
        """
        super(EllipticalSersic, self).__init__(centre, axis_ratio, phi, settings)
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @property
    def sersic_constant(self):
        """

        Returns
        -------
        sersic_constant: float
            A parameter, derived from sersic_index, that ensures that effective_radius always contains 50% of the light.
        """
        return (2 * self.sersic_index) - (1. / 3.) + (4. / (405. * self.sersic_index)) + (
                46. / (25515. * self.sersic_index ** 2)) + (131. / (1148175. * self.sersic_index ** 3)) - (
                       2194697. / (30690717750. * self.sersic_index ** 4))

    def intensity_from_grid(self, grid):
        return self.intensity_at_grid_radii(self.grid_to_eccentric_radii(grid))

    def grid_to_eccentric_radii(self, grid):
        return np.multiply(np.sqrt(self.axis_ratio),
                           np.sqrt(np.add(np.square(grid[:, 0]),
                                          np.square(np.divide(grid[:, 1], self.axis_ratio))))).view(np.ndarray)

    def intensity_at_grid_radii(self, grid_radii):
        return np.multiply(self.intensity, np.exp(
            np.multiply(-self.sersic_constant,
                        np.add(np.power(np.divide(grid_radii, self.effective_radius), 1. / self.sersic_index), -1))))


sub_grid_size = 4

lsst = profiling_data.setup_class(name='LSST', pixel_scale=0.2, sub_grid_size=sub_grid_size)
euclid = profiling_data.setup_class(name='Euclid', pixel_scale=0.1, sub_grid_size=sub_grid_size)
hst = profiling_data.setup_class(name='HST', pixel_scale=0.05, sub_grid_size=sub_grid_size)
hst_up = profiling_data.setup_class(name='HSTup', pixel_scale=0.03, sub_grid_size=sub_grid_size)
ao = profiling_data.setup_class(name='AO', pixel_scale=0.01, sub_grid_size=sub_grid_size)

sersic = EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, intensity=0.1,
                          effective_radius=0.8, sersic_index=4.0)


@tools.tick_toc_x10
def lsst_solution():
    sersic.intensity_from_grid(grid=lsst.coords.sub_grid_coords)


@tools.tick_toc_x10
def euclid_solution():
    sersic.intensity_from_grid(grid=euclid.coords.sub_grid_coords)


@tools.tick_toc_x10
def hst_solution():
    sersic.intensity_from_grid(grid=hst.coords.sub_grid_coords)


@tools.tick_toc_x10
def hst_up_solution():
    sersic.intensity_from_grid(grid=hst_up.coords.sub_grid_coords)


@tools.tick_toc_x10
def ao_solution():
    sersic.intensity_from_grid(grid=ao.coords.sub_grid_coords)


if __name__ == "__main__":
    lsst_solution()
    euclid_solution()
    hst_solution()
    hst_up_solution()
    ao_solution()
