from autolens.model.profiles.summary import profile_summary

from test.unit.mock.mock_summary import MockMassProfile, MockNFW, MockTruncatedNFW

import os

test_summary_dir = "{}/../../../test_files/summary/".format(os.path.dirname(os.path.realpath(__file__)))

class TestSummarizeMassProfile:

    def test__summarize_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_mass_profile(summary_file=summary_file, mass_profile=MockTruncatedNFW(),
                                               critical_surface_mass_density=1.0, cosmic_average_mass_density_arcsec=1.0,
                                               radii=[10.0, 500.0])

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text == ['Mass Profile = MockTruncatedNFW\n',
                                '\n',
                                'Einstein Radius = 10.00"\n',
                                'Mass within Einstein Radius = 1.0000e+03 solMass\n',
                                'Mass within 10.00" = 1.0000e+03 solMass\n',
                                'Mass within 500.00" = 1.0000e+03 solMass\n',
                                'Rho at scale radius = 100.00\n',
                                'Delta concentration = 200.00\n',
                                'Concentration = 300.00\n',
                                'Radius at 200x cosmic average density = 400.00"\n',
                                'Mass at 200x cosmic average density = 500.00 solMass\n',
                                'Mass at truncation radius = 600.00 solMass\n']

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')

    def test__summarize_einstein_radius_and_mass___multiple_radii_both_output(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_einstein_radius_and_mass(summary_file=summary_file, mass_profile=MockMassProfile(),
                                                           critical_surface_mass_density=1.0)

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text == ['Einstein Radius = 10.00"\n',
                                'Mass within Einstein Radius = 1.0000e+03 solMass\n']

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')

    def test__summarize_mass_within_radii___multiple_radii_both_output(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_mass_within_radii(summary_file=summary_file, mass_profile=MockMassProfile(),
                                                    radii=[10.0, 500.0], critical_surface_mass_density=1.0)

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text == ['Mass within 10.00" = 1.0000e+03 solMass\n',
                                'Mass within 500.00" = 1.0000e+03 solMass\n']

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')

    def test__summarize_nfw_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_nfw_mass_profile(summary_file=summary_file, nfw=MockNFW(),
                                                   critical_surface_mass_density_arcsec=1.0,
                                                   cosmic_average_mass_density_arcsec=1.0)

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text == ['Rho at scale radius = 100.00\n',
                                'Delta concentration = 200.00\n',
                                'Concentration = 300.00\n',
                                'Radius at 200x cosmic average density = 400.00"\n',
                                'Mass at 200x cosmic average density = 500.00 solMass\n']

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')

    def test__summarize_truncated_nfw_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_truncated_nfw_mass_profile(summary_file=summary_file, truncated_nfw=MockTruncatedNFW(),
                                                             critical_surface_mass_density_arcsec=1.0,
                                                             cosmic_average_mass_density_arcsec=1.0)

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text == ['Rho at scale radius = 100.00\n',
                                'Delta concentration = 200.00\n',
                                'Concentration = 300.00\n',
                                'Radius at 200x cosmic average density = 400.00"\n',
                                'Mass at 200x cosmic average density = 500.00 solMass\n',
                                'Mass at truncation radius = 600.00 solMass\n']

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')

    def test__summarize_truncated_nfw_challenge_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        profile_summary.summarize_truncated_nfw_challenge_mass_profile(summary_file=summary_file,
                                                                       truncated_nfw_challenge=MockTruncatedNFW())

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text == ['Rho at scale radius = 100.00\n',
                                'Delta concentration = 200.00\n',
                                'Concentration = 300.00\n',
                                'Radius at 200x cosmic average density = 400.00"\n',
                                'Mass at 200x cosmic average density = 500.00 solMass\n',
                                'Mass at truncation radius = 600.00 solMass\n']

        summary_file.close()

        os.remove(path=test_summary_dir + 'model.summary')