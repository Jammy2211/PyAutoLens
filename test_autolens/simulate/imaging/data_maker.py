from test_autolens.simulate.imaging import makers

# Welcome to the PyAutoLens test_autoarray suite data maker. Here, we'll make the suite of data_type that we use to test and profile
# PyAutoLens. This consists of the following sets of images:

# A lens only image, where the lens is an elliptical Dev Vaucouleurs profile.
# A lens only image, where the lens is a bulge (Dev Vaucouleurs) + Envelope (Exponential) profile.
# A lens only image, where there are two lens galaxies both composed of Sersic bules.
# A source-only image, where the lens mass is an SIE and the source light is a smooth Exponential.
# A source-only image, where the lens mass is an SIE and source light a cuspy Sersic (sersic_index=3).
# The same smooth source image above, but with an offset lens / source centre such that the lens / source galaxy images
# are not as the centre of the image.
# A lens + source image, where the lens light is a Dev Vaucouleurs, mass is an SIE and the source light is a smooth
# Exponential.
# A lens + source image, where the lens light is a Dev Vaucouleurs, mass is an SIE and source light a cuspy Sersic
# (sersic_index=3).

# Each image is generated at 5 resolutions, 0.2" (LSST), 0.1" (Euclid), 0.05" (HST), 0.03" (HST), 0.01" (Keck AO).

sub_size = 1
data_resolutions = ["lsst", "euclid", "hst", "hst_up", "ao"]

# To simulator each lens, we pass it a name and call its maker. In the makers.py file, you'll see the
makers.make_lens_light_dev_vaucouleurs(
    data_resolutions=data_resolutions, sub_size=sub_size
)
makers.make_lens_bulge_disk(data_resolutions=data_resolutions, sub_size=sub_size)
makers.make_lens_x2_light(data_resolutions=data_resolutions, sub_size=sub_size)
makers.make_lens_mass__source_smooth(
    data_resolutions=data_resolutions, sub_size=sub_size
)
makers.make_lens_mass__source_cuspy(
    data_resolutions=data_resolutions, sub_size=sub_size
)
makers.make_lens_sis__source_smooth(
    data_resolutions=data_resolutions, sub_size=sub_size
)
makers.make_lens_mass__source_smooth__offset_centre(
    data_resolutions=data_resolutions, sub_size=sub_size
)
makers.make_lens_light__source_smooth(
    data_resolutions=data_resolutions, sub_size=sub_size
)
makers.make_lens_light__source_cuspy(
    data_resolutions=data_resolutions, sub_size=sub_size
)
