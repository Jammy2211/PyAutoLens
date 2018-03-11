import os
from auto_lens import multi_nest_tools

data_path = "{}/".format(os.path.dirname(os.path.realpath(__file__)))

class TestGenerateLatex(object):

    def test__one_parameter__no_subscript(self):

        assert multi_nest_tools.generate_parameter_latex('x') == ['$x$']

    def test__three_parameters__no_subscript(self):

        assert multi_nest_tools.generate_parameter_latex(['x', 'y', 'z']) == ['$x$', '$y$', '$z$']

    def test__one_parameter__subscript__no_number(self):

        assert multi_nest_tools.generate_parameter_latex(['x'], subscript='d') == [r'$x_{\mathrm{d}}$']

    def test__three_parameters__subscript__no_number(self):

        assert multi_nest_tools.generate_parameter_latex(['x', 'y', 'z'], subscript='d') == [r'$x_{\mathrm{d}}$',
                                                                                    r'$y_{\mathrm{d}}$',
                                                                                    r'$z_{\mathrm{d}}$']

class TestLoadModels:

    def test__read_most_probable_vector__short_summary(self):

        most_probable_vector = multi_nest_tools.read_most_probable(results_path=
                                                                   data_path+'test_files/multinest/short_')

        assert most_probable_vector == [1.0, 2.0, 3.0, 4.0]

    def test__read_most_probable_vector__long_summary(self):

        most_probable_vector = multi_nest_tools.read_most_probable(results_path=
                                                                   data_path+'test_files/multinest/long_')

        assert most_probable_vector == [1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0, 11.0, 12.0]

    def test__read_most_likely_vector__short_summary(self):

        most_likely_vector = multi_nest_tools.read_most_likely(results_path=
                                                                   data_path+'test_files/multinest/short_')

        assert most_likely_vector == [5.0, 6.0, 7.0, 8.0]

    def test__read_most_likely_vector__long_summary(self):

        most_likely_vector = multi_nest_tools.read_most_likely(results_path=
                                                                   data_path+'test_files/multinest/long_')

        assert most_likely_vector == [13.0, 14.0, 15.0, 16.0, -17.0, -18.0, -19.0, -20.0, 21.0, 22.0, 23.0, 24.0]