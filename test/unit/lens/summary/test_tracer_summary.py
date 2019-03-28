from autolens.lens.summary import tracer_summary

from test.unit.mock.mock_summary import MockTracer, MockPlane, MockGalaxy, MockTruncatedNFW

import os

test_summary_dir = "{}/../../test_files/summary/".format(os.path.dirname(os.path.realpath(__file__)))

class TestSummarizePlane:

    def test__summarize_tracer_with_galaxy_with_nfw_mass_profile(self):

        summary_file = open(file=test_summary_dir + 'model.summary', mode="w+")

        mock_plane = MockPlane(galaxies=[MockGalaxy(mass_profiles=[MockTruncatedNFW()])])

        tracer_summary.summarize_tracer(summary_file=summary_file, tracer=MockTracer(planes=[mock_plane]),
                                        radii=[10.0, 500.0])

        summary_file.close()

        summary_file_read = open(test_summary_dir + 'model.summary', mode="r")
        summary_text = summary_file_read.readlines()

        assert summary_text == ['Tracer Cosmology = FlatLambdaCDM(name="Planck15", H0=67.7 km / (Mpc s), Om0=0.307, Tcmb0=2.725 K, Neff=3.05, m_nu=[0.   0.   0.06] eV, Ob0=0.0486)\n',
                                'Tracer Redshifts = [0.5]\n',
                                '\n',
                                'Plane Redshift = 0.5\n',
                                'Plane Critical Surface Mass Density (solMass / arcsec^2) = 2.0\n',
                                '\n',
                                'Galaxy = lol\n',
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