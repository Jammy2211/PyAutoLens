from autolens import text_util


def test__within_radius_label_value_and_unit_string():
    string0 = text_util.within_radius_label_value_and_unit_string(
        prefix='mass', radius=1.0, unit_length='arcsec', value=30.0,
        unit_value='solMass', whitespace=40)

    string1 = text_util.within_radius_label_value_and_unit_string(
        prefix='mass', radius=1.0, unit_length='arcsec', value=30.0,
        unit_value='solMass', whitespace=35)

    string2 = text_util.within_radius_label_value_and_unit_string(
        prefix='mass', radius=1.0, unit_length='arcsec', value=30.0,
        unit_value='solMass', whitespace=30)

    assert string0 == 'mass_within_1.00_arcsec                 3.0000e+01 solMass'
    assert string1 == 'mass_within_1.00_arcsec            3.0000e+01 solMass'
    assert string2 == 'mass_within_1.00_arcsec       3.0000e+01 solMass'

    string = text_util.within_radius_label_value_and_unit_string(
        prefix='mass', radius=1.0, unit_length='arcsec2', value=40.0,
        unit_value='solMass2', whitespace=40)

    assert string == 'mass_within_1.00_arcsec2                4.0000e+01 solMass2'
