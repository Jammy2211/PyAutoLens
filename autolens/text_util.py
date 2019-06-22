from autofit.tools import text_util


def within_radius_label_value_and_unit_string(prefix, radius, unit_length, value,
                                              unit_value, whitespace):
    label = prefix + '_within_{:.2f}_{}'.format(radius, unit_length)
    return text_util.label_value_and_unit_string(
        label=label, value=value, unit=unit_value, whitespace=whitespace)
