import autolens as al

from test_autolens.simulators.imaging import instrument_util


def simulate__mass_sie__source_sersic(instrument):

    dataset_name = "mass_sie__source_sersic"

    # This source-only system has a smooth source (low Sersic Index) and simple SIE mass profile.

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            elliptical_comps=(-0.055555, 0.096225),
            intensity=0.4,
            effective_radius=0.5,
            sersic_index=1.0,
        ),
    )

    instrument_util.simulate_imaging_from_instrument(
        instrument=instrument,
        dataset_name=dataset_name,
        galaxies=[lens_galaxy, source_galaxy],
    )


def simulate__mass_sie__source_sersic__offset_centre(instrument):

    dataset_name = "mass_sie__source_sersic__offset_centre"

    # This source-only system has a smooth source (low Sersic Index) and simple SIE mass profile.

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.SphericalIsothermal(centre=(2.0, 2.0), einstein_radius=1.2),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(2.0, 2.0),
            elliptical_comps=(-0.055555, 0.096225),
            intensity=0.4,
            effective_radius=0.5,
            sersic_index=1.0,
        ),
    )

    instrument_util.simulate_imaging_from_instrument(
        instrument=instrument,
        dataset_name=dataset_name,
        galaxies=[lens_galaxy, source_galaxy],
    )


def simulate__light_sersic__source_sersic(instrument):

    dataset_name = "light_sersic__source_sersic"

    # This source-only system has a smooth source (low Sersic Index) and simple SIE mass profile.

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            elliptical_comps=(0.1, 0.0),
            intensity=0.5,
            effective_radius=0.8,
            sersic_index=4.0,
        ),
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            elliptical_comps=(-0.055555, 0.096225),
            intensity=0.4,
            effective_radius=0.5,
            sersic_index=1.0,
        ),
    )

    instrument_util.simulate_imaging_from_instrument(
        instrument=instrument,
        dataset_name=dataset_name,
        galaxies=[lens_galaxy, source_galaxy],
    )
