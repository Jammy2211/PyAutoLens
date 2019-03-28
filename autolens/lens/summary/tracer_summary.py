from autolens.lens.summary import plane_summary

def summarize_tracer(summary_file, tracer, radii):

    # TODO : There will be methods here which summarize overall plane properties (e.g. total mass for all galaxies
    # TODO : redshifts. For now, we'll just iterate over all mass profiles and output the mass profile info.
    # TODO : Also not sure how to access the Galaxy name...

    summary_file.write('Tracer Cosmology = ' + str(tracer.cosmology) + '\n')
    summary_file.write('Tracer Redshifts = ' + str(tracer.plane_redshifts) + '\n\n')

    for plane in tracer.planes:

        plane_summary.summarize_plane(summary_file=summary_file, plane=plane,
                                      critical_surface_mass_density=tracer.critical_surface_mass_density_arcsec,
                                      radii=radii)