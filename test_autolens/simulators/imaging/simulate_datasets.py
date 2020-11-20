from test_autolens.simulators.imaging import simulators

"""
Welcome to the PyAutoLens dataset generator. Here, we'll make the datasets used to test and profile PyAutoLens.

This consists of the following sets of images:

- A source-only image, where the lens mass is an SIE and the source light is a smooth Exponential.
- A source-only image, where the lens mass is an SIE and source light a cuspy Sersic (sersic_index=3).
- The same smooth source image above, but with an offset lens / source centre such that the 
  lens / source galaxy images are not as the centre of the image.
- A lens + source image, where the lens light is a Dev Vaucouleurs, mass is an SIE and the source light is a smooth
  Exponential.
- A lens + source image, where the lens light is a Dev Vaucouleurs, mass is an SIE and source light a cuspy Sersic
  (sersic_index=3).

Each image is generated for up to 5 instruments, VRO, Euclid, HST, HST, Keck AO.
"""

# To simulate each lens, we pass it an instrument and call its simulate function.

instruments = ["vro"]  # , "euclid", "hst", "hst_up", "ao"]

for instrument in instruments:

    simulators.simulate__mass_sie__source_sersic(instrument=instrument)
    simulators.simulate__mass_sie__source_sersic__offset_centre(instrument=instrument)
    simulators.simulate__light_sersic__source_sersic(instrument=instrument)
