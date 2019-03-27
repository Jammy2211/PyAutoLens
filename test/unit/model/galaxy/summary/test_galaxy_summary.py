from autolens.model.galaxy.summary import galaxy_summary

from test.unit.mock.mock_summary import MockGalaxy, MockTruncatedNFW

import os

test_summary_dir = "{}/../../../test_files/summary/".format(os.path.dirname(os.path.realpath(__file__)))

class TestSummarizeGalaxy:

    def test__summarize_galaxy_with_nfw_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        galaxy_summary.summarize_galaxy(summary_file=summary_file, galaxy=MockGalaxy(mass_profiles=[MockTruncatedNFW()]),
                                        critical_surface_mass_density=1.0, cosmic_average_mass_density_arcsec=1.0,
                                        radii=[10.0, 500.0])

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text == ['Galaxy = lol\n',
                                '\n',
                                'Mass Profile = MockTruncatedNFW\n',
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