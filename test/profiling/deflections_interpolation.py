import time

from autolens.model.profiles import mass_profiles as mp
from autolens.lens import lens_data as ld
from autolens.array import mask as msk

from test.simulation import simulation_util

# Although we could test the deflection angles without using an image (e.al. by just making a grid), we have chosen to
# set this test up using an image and mask. This gives run-time numbers that can be easily related to an actual lens
# analysis

sub_size = 4
radius_arcsec = 4.0

print("sub grid size = " + str(sub_size))
print("circular mask radius = " + str(radius_arcsec) + "\n")

for data_resolution in ["LSST", "Euclid", "HST", "HST_Up", "AO"]:

    imaging_data = simulation_util.load_test_imaging_data(
        data_type="lens_mass__source_smooth",
        data_resolution=data_resolution,
        psf_shape=(3, 3),
    )
    mask = al.Mask.circular(
        shape=imaging_data.shape,
        pixel_scale=imaging_data.pixel_scale,
        radius_arcsec=radius_arcsec,
    )
    lens_data = al.LensData(
        imaging_data=imaging_data,
        mask=mask,
        sub_size=sub_size,
        pixel_scale_interpolation_grid=0.05,
    )

    print("Deflection angle run times for image type " + data_resolution + "\n")
    print("Number of points = " + str(lens_data.grid.shape[0]) + "\n")

    print(
        "Interpolation Pixel Scale = "
        + str(lens_data.pixel_scale_interpolation_grid)
        + "\n"
    )
    print(
        "Number of Interpolation Points "
        + str(lens_data.grid.interpolator.interp_grid.shape[0])
        + "\n"
    )

    ### EllipticalIsothermal ###

    mass_profile = al.EllipticalIsothermal(
        centre=(0.001, 0.001), axis_ratio=0.8, phi=45.0, einstein_radius=1.0
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalIsothermal time = {}".format(diff))

    ### SphericalIsothermal ###

    mass_profile = al.mass_profiles.SphericalIsothermal(
        centre=(0.001, 0.001), einstein_radius=1.0
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalIsothermal time = {}".format(diff))

    ### EllipticalPowerLaw (slope = 1.5) ###

    mass_profile = al.EllipticalPowerLaw(
        centre=(0.001, 0.001), axis_ratio=0.8, phi=45.0, einstein_radius=1.0, slope=1.5
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalPowerLaw (slope = 1.5) time = {}".format(diff))

    ### SphericalPowerLaw (slope = 1.5) ###

    mass_profile = al.SphericalPowerLaw(
        centre=(0.001, 0.001), einstein_radius=1.0, slope=1.5
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalPowerLaw (slope = 1.5) time = {}".format(diff))

    ### EllipticalPowerLaw (slope = 2.5) ###

    mass_profile = al.EllipticalPowerLaw(
        centre=(0.001, 0.001), axis_ratio=0.8, phi=45.0, einstein_radius=1.0, slope=2.5
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalPowerLaw (slope = 2.5) time = {}".format(diff))

    ### SphericalPowerLaw (slope = 2.5) ###

    mass_profile = al.SphericalPowerLaw(
        centre=(0.001, 0.001), einstein_radius=1.0, slope=2.5
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalPowerLaw (slope = 2.5) time = {}".format(diff))

    ### EllipticalCoredPowerLaw ###

    mass_profile = al.EllipticalCoredPowerLaw(
        centre=(0.001, 0.001),
        axis_ratio=0.8,
        phi=45.0,
        einstein_radius=1.0,
        slope=2.0,
        core_radius=0.1,
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalCoredPowerLaw time = {}".format(diff))

    ### SphericalCoredPowerLaw ###

    mass_profile = al.SphericalCoredPowerLaw(
        centre=(0.001, 0.001), einstein_radius=1.0, slope=2.0, core_radius=0.1
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalCoredPowerLaw time = {}".format(diff))

    ### EllipticalGeneralizedNFW (inner_slope = 0.5) ###

    # mass_profile = al.EllipticalGeneralizedNFW(centre=(0.001, 0.001), axis_ratio=0.8, phi=45.0, kappa_s=0.1,
    #                                            scale_radius=10.0, inner_slope=0.5)
    #
    # start = time.time()
    # mass_profile.deflections_from_grid(grid=lens_data.grid)
    # diff = time.time() - start
    # print("EllipticalGeneralizedNFW (inner_slope = 1.0) time = {}".format(diff))

    ### SphericalGeneralizedNFW (inner_slope = 0.5) ###

    mass_profile = al.SphericalGeneralizedNFW(
        centre=(0.001, 0.001), kappa_s=0.1, scale_radius=10.0, inner_slope=0.5
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalGeneralizedNFW (inner_slope = 1.0) time = {}".format(diff))

    ### EllipticalNFW ###

    mass_profile = al.EllipticalNFW(
        centre=(0.001, 0.001), axis_ratio=0.8, phi=45.0, kappa_s=0.1, scale_radius=10.0
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalNFW time = {}".format(diff))

    ### SphericalNFW ###

    mass_profile = al.SphericalNFW(
        centre=(0.001, 0.001), kappa_s=0.1, scale_radius=10.0
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalNFW time = {}".format(diff))

    ### EllipticalExponential ###

    profile = al.EllipticalExponential(
        centre=(0.001, 0.001),
        axis_ratio=0.8,
        phi=45.0,
        intensity=1.0,
        effective_radius=1.0,
        mass_to_light_ratio=1.0,
    )

    start = time.time()
    profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalExponential time = {}".format(diff))

    ### SphericalExponential ###

    profile = al.SphericalExponential(
        centre=(0.001, 0.001),
        intensity=1.0,
        effective_radius=1.0,
        mass_to_light_ratio=1.0,
    )

    start = time.time()
    profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalExponential time = {}".format(diff))

    ### EllipticalDevVaucouleurs ###

    profile = al.EllipticalDevVaucouleurs(
        centre=(0.001, 0.001),
        axis_ratio=0.8,
        phi=45.0,
        intensity=1.0,
        effective_radius=1.0,
        mass_to_light_ratio=1.0,
    )

    start = time.time()
    profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalDevVaucouleurs time = {}".format(diff))

    ### SphericalDevVaucouleurs ###

    profile = al.SphericalDevVaucouleurs(
        centre=(0.001, 0.001),
        intensity=1.0,
        effective_radius=1.0,
        mass_to_light_ratio=1.0,
    )

    start = time.time()
    profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalDevVaucouleurs time = {}".format(diff))

    ### EllipticalSersic ###

    mass_profile = al.EllipticalSersic(
        centre=(0.001, 0.001),
        axis_ratio=0.8,
        phi=45.0,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
        mass_to_light_ratio=1.0,
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalSersic time = {}".format(diff))

    ### SphericalSersic ###

    mass_profile = al.SphericalSersic(
        centre=(0.001, 0.001),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
        mass_to_light_ratio=1.0,
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalSersic time = {}".format(diff))

    ### EllipticalSersicRadialGradient (gradient = -1.0) ###

    mass_profile = al.EllipticalSersicRadialGradient(
        centre=(0.001, 0.001),
        axis_ratio=0.8,
        phi=45.0,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=-1.0,
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalSersicRadialGradient (gradient = -1.0) time = {}".format(diff))

    ### SphericalersicRadialGradient (gradient = -1.0) ###

    mass_profile = al.SphericalSersicRadialGradient(
        centre=(0.001, 0.001),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=-1.0,
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalSersicRadialGradient (gradient = -1.0) time = {}".format(diff))

    ### EllipticalSersicRadialGradient (gradient = 1.0) ###

    mass_profile = al.EllipticalSersicRadialGradient(
        centre=(0.001, 0.001),
        axis_ratio=0.8,
        phi=45.0,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalSersicRadialGradient (gradient = 1.0) time = {}".format(diff))

    ### SphericalersicRadialGradient (gradient = 1.0) ###

    mass_profile = al.SphericalSersicRadialGradient(
        centre=(0.001, 0.001),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    start = time.time()
    mass_profile.deflections_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalSersicRadialGradient (gradient = 1.0) time = {}".format(diff))

    print()
