from autolens.model.profiles.summary import profile_summary

def summarize_galaxy(summary_file, galaxy, critical_surface_mass_density, cosmic_average_mass_density_arcsec, radii):

    # TODO : There will be methods here which summarize overall galaxy properties (e.g. total mass for all mass
    # TODO : profiles, redshifts. For now, we'll just iterate over all mass profiles and output the mass profile info.
    # TODO : Also not sure how to access the Galaxy name...

    summary_file.write('Galaxy = lol\n\n')# + galaxy.name)

    for mass_profile in galaxy.mass_profiles:

        profile_summary.summarize_mass_profile(summary_file=summary_file, mass_profile=mass_profile,
                                               critical_surface_mass_density=critical_surface_mass_density,
                                               cosmic_average_mass_density_arcsec=cosmic_average_mass_density_arcsec,
                                               radii=radii)