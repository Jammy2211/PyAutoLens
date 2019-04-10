from autolens.plotters import plotter_util

class TestRadii:

    def test__radii_bin_size_from_minimum_and_maximum_radii__is_correct_values(self):

        radii_bin_size = plotter_util.radii_bin_size_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=0.0, maximum_radius=1.0, radii_bins=1)

        assert radii_bin_size == 1.0

        radii_bin_size = plotter_util.radii_bin_size_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=0.0, maximum_radius=1.0, radii_bins=2)

        assert radii_bin_size == 0.5

        radii_bin_size = plotter_util.radii_bin_size_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=0.0, maximum_radius=2.0, radii_bins=1)

        assert radii_bin_size == 2.0

        radii_bin_size = plotter_util.radii_bin_size_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=1.0, maximum_radius=6.0, radii_bins=5)

        assert radii_bin_size == 1.0

        radii_bin_size = plotter_util.radii_bin_size_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=100.0, maximum_radius=200.0, radii_bins=25)

        assert radii_bin_size == 4.0

    def test__quantity_radii_from_minimum_and_maximum_radii__is_correct_values(self):

        quantity_radii = plotter_util.quantity_radii_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=0.0, maximum_radius=1.0, radii_bins=1)

        assert quantity_radii == [0.0, 1.0]

        quantity_radii = plotter_util.quantity_radii_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=0.0, maximum_radius=1.0, radii_bins=2)

        assert quantity_radii == [0.0, 0.5, 1.0]

        quantity_radii = plotter_util.quantity_radii_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=0.0, maximum_radius=1.0, radii_bins=4)

        assert quantity_radii == [0.0, 0.25, 0.5, 0.75, 1.0]

        quantity_radii = plotter_util.quantity_radii_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=0.0, maximum_radius=2.0, radii_bins=4)

        assert quantity_radii == [0.0, 0.5, 1.0, 1.5, 2.0]

        quantity_radii = plotter_util.quantity_radii_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=1.0, maximum_radius=3.0, radii_bins=4)

        assert quantity_radii == [1.0, 1.5, 2.0, 2.5, 3.0]

    def test__annuli_radii_from_minimum_and_maximum_radii__is_correct_values(self):

        annuli_radii = plotter_util.annuli_radii_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=0.0, maximum_radius=1.0, radii_bins=1)

        assert (annuli_radii == [-0.5, 0.5, 1.5]).all()

        annuli_radii = plotter_util.annuli_radii_from_minimum_and_maximum_radii_and_radii_bins(
            minimum_radius=0.0, maximum_radius=1.0, radii_bins=2)


        assert (annuli_radii == [-0.5, 0.0, 0.5, 1.0, 1.5]).all()
