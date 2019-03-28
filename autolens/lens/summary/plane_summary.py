from autolens.model.galaxy.summary import galaxy_summary

def summarize_plane(summary_file, plane, critical_surface_mass_density, radii):

    # TODO : There will be methods here which summarize overall plane properties (e.g. total mass for all galaxies
    # TODO : redshifts. For now, we'll just iterate over all mass profiles and output the mass profile info.
    # TODO : Also not sure how to access the Galaxy name...

    summary_file.write('Plane Redshift = ' + str(plane.redshift) + '\n')
    summary_file.write('Plane Critical Surface Mass Density (solMass / arcsec^2) = ' + str(critical_surface_mass_density) + '\n\n')

    for galaxy in plane.galaxies:

        galaxy_summary.summarize_galaxy(summary_file=summary_file, galaxy=galaxy,
                                        critical_surface_mass_density=critical_surface_mass_density,
                                        cosmic_average_mass_density_arcsec=plane.cosmic_average_mass_density_arcsec,
                                        radii=radii)