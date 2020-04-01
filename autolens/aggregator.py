import autolens as al

def make_tracer(output):

    output = output.output

    return al.Tracer.from_galaxies(
        galaxies=output.most_likely_instance.galaxies
    )
