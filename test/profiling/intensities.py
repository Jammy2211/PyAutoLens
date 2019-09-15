import time

from autolens.model.profiles import light_profiles as lp
from autolens.array import mask as msk
from autolens.lens import lens_data as ld

from test.simulation import simulation_util

# Although we could test the image without using an image (e.al. by just making a grid), we have chosen to
# set this test up using an image and mask. This gives run-time numbers that can be easily related to an actual lens
# analysis

sub_size = 4
radius_arcsec = 3.0

print("sub grid size = " + str(sub_size))
print("circular mask radius = " + str(radius_arcsec) + "\n")

for data_resolution in ["LSST", "Euclid", "HST", "HST_Up", "AO"]:

    imaging_data = simulation_util.load_test_imaging_data(
        data_type="no_lens_source_smooth",
        data_resolution=data_resolution,
        psf_shape=(3, 3),
    )
    mask = al.Mask.circular(
        shape=imaging_data.shape,
        pixel_scale=imaging_data.pixel_scale,
        radius_arcsec=radius_arcsec,
    )
    lens_data = al.LensData(imaging_data=imaging_data, mask=mask, sub_size=sub_size)

    print("Deflection angle run times for image type " + data_resolution + "\n")
    print("Number of points = " + str(lens_data.grid.shape[0]) + "\n")

    ### EllipticalGaussian ###

    mass_profile = al.EllipticalGaussian(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, sigma=1.0
    )

    start = time.time()
    mass_profile.profile_image_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalGaussian time = {}".format(diff))

    ### SphericalGaussian ###

    mass_profile = al.SphericalGaussian(centre=(0.0, 0.0), sigma=1.0)

    start = time.time()
    mass_profile.profile_image_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalGaussian time = {}".format(diff))

    ### EllipticalExponential ###

    profile = al.EllipticalExponential(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, intensity=1.0, effective_radius=1.0
    )

    start = time.time()
    profile.profile_image_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalExponential time = {}".format(diff))

    ### SphericalExponential ###

    profile = al.SphericalExponential(
        centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0
    )

    start = time.time()
    profile.profile_image_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalExponential time = {}".format(diff))

    ### EllipticalDevVaucouleurs ###

    profile = al.EllipticalDevVaucouleurs(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, intensity=1.0, effective_radius=1.0
    )

    start = time.time()
    profile.profile_image_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalDevVaucouleurs time = {}".format(diff))

    ### SphericalDevVaucouleurs ###

    profile = al.SphericalDevVaucouleurs(
        centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0
    )

    start = time.time()
    profile.profile_image_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalDevVaucouleurs time = {}".format(diff))

    ### EllipticalSersic ###

    mass_profile = al.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.8,
        phi=45.0,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    )

    start = time.time()
    mass_profile.profile_image_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalSersic time = {}".format(diff))

    ### SphericalSersic ###

    mass_profile = al.light_profiles.SphericalSersic(
        centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=2.5
    )

    start = time.time()
    mass_profile.profile_image_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalSersic time = {}".format(diff))

    ### EllipticalCoreSersic ###

    mass_profile = al.EllipticalCoreSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.8,
        phi=45.0,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
        radius_break=0.01,
        intensity_break=0.05,
        gamma=0.25,
        alpha=3.0,
    )

    start = time.time()
    mass_profile.profile_image_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("EllipticalCoreSersic time = {}".format(diff))

    ### SphericalCoreSersic ###

    mass_profile = al.SphericalCoreSersic(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
        radius_break=0.01,
        intensity_break=0.05,
        gamma=0.25,
        alpha=3.0,
    )

    start = time.time()
    mass_profile.profile_image_from_grid(grid=lens_data.grid)
    diff = time.time() - start
    print("SphericalCoreSersic time = {}".format(diff))

    print()
