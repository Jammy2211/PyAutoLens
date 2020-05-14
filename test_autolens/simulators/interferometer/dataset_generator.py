from test_autolens.simulators.interferometer import simulators

"""
Welcome to the PyAutoLens dataset generator. Here, we'll make the datasets that we use to test and profile
PyAutoLens. This consists of the following sets of images:

- A lens only image, where the lens is an elliptical Dev Vaucouleurs profile.
- A lens only image, where the lens is a bulge (Dev Vaucouleurs) + Envelope (Exponential) profile.
- A lens only image, where there are two lens galaxies both composed of Sersic bules.
- A source-only image, where the lens mass is an SIE and the source light is a smooth Exponential.
- A source-only image, where the lens mass is an SIE and source light a cuspy Sersic (sersic_index=3).
- The same smooth source image above, but with an offset lens / source centre such that the lens / source galaxy images
are not as the centre of the image.
- A lens + source image, where the lens light is a Dev Vaucouleurs, mass is an SIE and the source light is a smooth
Exponential.
- A lens + source image, where the lens light is a Dev Vaucouleurs, mass is an SIE and source light a cuspy Sersic
(sersic_index=3).

Each image is generated for one instrument.
"""

instruments = ["sma"]

# To simulate each lens, we pass it a name and call its maker. In the simulators.py file, you'll see the
simulators.simulate_lens_light_dev_vaucouleurs(instruments=instruments)
simulators.simulate_lens_bulge_disk(instruments=instruments)
simulators.simulate_lens_x2_light(instruments=instruments)
simulators.simulate_lens_sie__source_smooth(instruments=instruments)
simulators.simulate_lens_sie__source_cuspy(instruments=instruments)
simulators.simulate_lens_sis__source_smooth(instruments=instruments)
simulators.simulate_lens_sie__source_smooth__offset_centre(instruments=instruments)
simulators.simulate_lens_light__source_smooth(instruments=instruments)
simulators.simulate_lens_light__source_cuspy(instruments=instruments)
