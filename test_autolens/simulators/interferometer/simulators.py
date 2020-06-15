import autolens as al

from test_autolens.simulators.interferometer import instrument_util


def simulate__lens_sie__source_smooth(instrument):

    data_name = "lens_sie__source_smooth"

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
            e1=-0.055555,
            e2=0.096225,
            intensity=0.4,
            effective_radius=0.5,
            sersic_index=1.0,
        ),
    )

    instrument_util.simulate_interferometer_from_instrument(
        instrument=instrument,
        data_name=data_name,
        galaxies=[lens_galaxy, source_galaxy],
    )


def simulate__lens_sie__source_cuspy(instrument):

    data_name = "lens_sie__source_cuspy"

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
            e1=-0.055555,
            e2=0.096225,
            intensity=0.1,
            effective_radius=0.5,
            sersic_index=3.0,
        ),
    )

    instrument_util.simulate_interferometer_from_instrument(
        instrument=instrument,
        data_name=data_name,
        galaxies=[lens_galaxy, source_galaxy],
    )


def simulate__lens_sis__source_smooth(instrument):

    data_name = "lens_sis__source_smooth"

    # This source-only system has a smooth source (low Sersic Index) and simple SIE mass profile.

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            e1=-0.055555,
            e2=0.096225,
            intensity=0.4,
            effective_radius=0.5,
            sersic_index=1.0,
        ),
    )

    instrument_util.simulate_interferometer_from_instrument(
        instrument=instrument,
        data_name=data_name,
        galaxies=[lens_galaxy, source_galaxy],
    )


def simulate__lens_sie__source_smooth__offset_centre(instrument):

    data_name = "lens_sie__source_smooth__offset_centre"

    # This source-only system has a smooth source (low Sersic Index) and simple SIE mass profile.

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.SphericalIsothermal(centre=(2.0, 2.0), einstein_radius=1.2),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(2.0, 2.0),
            e1=-0.055555,
            e2=0.096225,
            intensity=0.4,
            effective_radius=0.5,
            sersic_index=1.0,
        ),
    )

    instrument_util.simulate_interferometer_from_instrument(
        instrument=instrument,
        data_name=data_name,
        galaxies=[lens_galaxy, source_galaxy],
    )


def simulate__lens_light__source_smooth(instrument):

    data_name = "lens_light__source_smooth"

    # This source-only system has a smooth source (low Sersic Index) and simple SIE mass profile.

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.9,
            phi=45.0,
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
            e1=-0.055555,
            e2=0.096225,
            intensity=0.4,
            effective_radius=0.5,
            sersic_index=1.0,
        ),
    )

    instrument_util.simulate_interferometer_from_instrument(
        instrument=instrument,
        data_name=data_name,
        galaxies=[lens_galaxy, source_galaxy],
    )


def simulate__lens_light__source_cuspy(instrument):

    data_name = "lens_light__source_cuspy"

    # This source-only system has a smooth source (low Sersic Index) and simple SIE mass profile.

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.9,
            phi=45.0,
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
            e1=-0.055555,
            e2=0.096225,
            intensity=0.1,
            effective_radius=0.5,
            sersic_index=3.0,
        ),
    )

    instrument_util.simulate_interferometer_from_instrument(
        instrument=instrument,
        data_name=data_name,
        galaxies=[lens_galaxy, source_galaxy],
    )
