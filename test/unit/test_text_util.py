from autolens import text_util

def test__within_radius_label_value_and_unit_string():

    string0 = text_util.within_radius_label_value_and_unit_string(
        prefix='pre_', radius=1.0, unit_length='arcsec', value=30.0, unit_value='solMass', whitespace=40,
        format_str_value='{:.4f}')

    string1 = text_util.within_radius_label_value_and_unit_string(
        prefix='pre_', radius=1.0, unit_length='arcsec', value=30.0, unit_value='solMass', whitespace=35,
        format_str_value='{:.4f}')

    string2 = text_util.within_radius_label_value_and_unit_string(
        prefix='pre_', radius=1.0, unit_length='arcsec', value=30.0, unit_value='solMass', whitespace=30,
        format_str_value='{:.4f}')

    assert string0 == 'pre__within_1.00_arcsec                 30.0000 solMass'
    assert string1 == 'pre__within_1.00_arcsec            30.0000 solMass'
    assert string2 == 'pre__within_1.00_arcsec       30.0000 solMass'

    string = text_util.within_radius_label_value_and_unit_string(
        prefix='pre222_', radius=1.0, unit_length='arcsec2', value=40.0, unit_value='solMass2', whitespace=40,
        format_str_value='{:.2f}')

    assert string == 'pre222__within_1.00_arcsec2             40.00 solMass2'
