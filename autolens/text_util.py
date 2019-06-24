import autofit as af


def within_radius_label_value_and_unit_string(prefix, radius, unit_length, value,
                                              unit_value, whitespace):
    label = prefix + '_within_{:.2f}_{}'.format(radius, unit_length)
    return af.text_util.label_value_and_unit_string(
        label=label, value=value, unit=unit_value, whitespace=whitespace)
