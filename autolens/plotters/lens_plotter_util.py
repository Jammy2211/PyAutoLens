

def get_unit_label_and_unit_conversion_factor(obj, plot_in_kpc):

    if plot_in_kpc:

        unit_label = 'kpc'
        unit_conversion_factor = obj.kpc_per_arcsec

    else:

        unit_label = 'arcsec'
        unit_conversion_factor = None

    return unit_label, unit_conversion_factor


def get_critical_curve_and_caustic(obj, include_critical_curves, include_caustics):

    if obj.has_mass_profile:

        if include_critical_curves:
            critical_curves = obj.critical_curves
        else:
            critical_curves = []

        if include_caustics:
            caustics = obj.caustics
        else:
            caustics = []

        return [critical_curves, caustics]

    else:

        return None