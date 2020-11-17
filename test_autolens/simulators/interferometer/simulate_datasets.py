from test_autolens.simulators.interferometer import simulators

"""
Welcome to the PyAutoLens dataset generator. Here, we'll make the datasets that we use to test and profile
PyAutoLens. This consists of the following sets of visibilitiess:

- Source-only visibilities, where the lens mass is an SIE and the source light is a smooth Exponential.
- Source-only visibilities, where the lens mass is an SIE and source light a cuspy Sersic (sersic_index=3).
- The same smooth source visibilities above, but with an offset lens / source centre such that the 
  lens / source galaxy visibilitiess are not as the centre of the visibilities.
- Lens + source visibilities, where the lens light is a Dev Vaucouleurs, mass is an SIE and the source light is a smooth
  Exponential.
- Lens + source visibilities, where the lens light is a Dev Vaucouleurs, mass is an SIE and source light a cuspy Sersic
  (sersic_index=3).

Each visibilities is generated for one instrument.
"""

# To simulate each lens, we pass it an instrument and call its simulate function.

instruments = ["sma"]

for instrument in instruments:

    simulators.simulate__mass_sie__source_sersic(instrument=instrument)
    simulators.simulate__mass_sie__source_sersic(instrument=instrument)
    simulators.simulate__mass_sie__source_sersic__offset_centre(instrument=instrument)
    simulators.simulate__light_sersic__source_sersic(instrument=instrument)
    simulators.simulate__light_sersic__source_sersic(instrument=instrument)
