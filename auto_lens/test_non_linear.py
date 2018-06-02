import os
import shutil
import pytest
from itertools import count
from auto_lens import non_linear
from auto_lens import model_mapper
from auto_lens.profiles import light_profiles, mass_profiles

path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))

def create_summary_4_parameters(path):

    summary = open(path + 'summary.txt', 'w')
    summary.write('    0.100000000000000000E+01   -0.200000000000000000E+01    0.300000000000000000E+01'
                  '    0.400000000000000000E+01   -0.500000000000000000E+01    0.600000000000000000E+01'
                  '    0.700000000000000000E+01    0.800000000000000000E+01    0.020000000000000000E+00'
                  '    0.999999990000000000E+07\n')
    summary.write('    0.100000000000000000E+01   -0.200000000000000000E+01    0.300000000000000000E+01'
                  '    0.400000000000000000E+01   -0.500000000000000000E+01    0.600000000000000000E+01'
                  '    0.700000000000000000E+01    0.800000000000000000E+01')
    summary.close()

def create_summary_11_parameters(path):

    summary = open(path + 'summary.txt', 'w')
    summary.write('    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01'
                  '    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01'
                  '   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01'
                  '    1.000000000000000000E+01    1.100000000000000000E+01    1.200000000000000000E+01'
                  '    1.300000000000000000E+01    1.400000000000000000E+01    1.500000000000000000E+01'
                  '    1.600000000000000000E+01   -1.700000000000000000E+01   -1.800000000000000000E+01'
                  '   -1.900000000000000000E+01   -2.000000000000000000E+01    2.100000000000000000E+01'
                  '    2.200000000000000000E+01    0.020000000000000000E+00    0.999999990000000000E+07\n')
    summary.write('    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01'
                  '    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01'
                  '   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01'
                  '    1.000000000000000000E+01    1.100000000000000000E+01    1.200000000000000000E+01'
                  '    1.300000000000000000E+01    1.400000000000000000E+01    1.500000000000000000E+01'
                  '    1.600000000000000000E+01   -1.700000000000000000E+01   -1.800000000000000000E+01'
                  '   -1.900000000000000000E+01   -2.000000000000000000E+01    2.100000000000000000E+01'
                  '    2.200000000000000000E+01')
    summary.close()

def create_weighted_samples_4_parameters(path):
    
    weighted_samples = open(path + 'obj.txt', 'w')
    weighted_samples.write('    0.020000000000000000E+00    0.999999990000000000E+07    0.110000000000000000E+01    0.210000000000000000E+01    0.310000000000000000E+01    0.410000000000000000E+01\n'
                           '    0.020000000000000000E+00    0.999999990000000000E+07    0.090000000000000000E+01    0.190000000000000000E+01    0.290000000000000000E+01    0.390000000000000000E+01\n'
                           '    0.010000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
                           '    0.050000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
                           '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
                           '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
                           '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
                           '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
                           '    0.200000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01\n'
                           '    0.300000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01')
    weighted_samples.close()

def create_weighted_samples_11_parameters(path):
    weighted_samples = open(path + 'obj.txt', 'w')
    weighted_samples.write('    0.020000000000000000E+00    0.999999990000000000E+07    0.110000000000000000E+01    0.210000000000000000E+01    0.310000000000000000E+01    0.410000000000000000E+01   -0.510000000000000000E+01   -0.610000000000000000E+01   -0.710000000000000000E+01   -0.810000000000000000E+01    0.910000000000000000E+01    1.010000000000000000E+01    1.110000000000000000E+01\n'
                           '    0.020000000000000000E+00    0.999999990000000000E+07    0.090000000000000000E+01    0.190000000000000000E+01    0.290000000000000000E+01    0.390000000000000000E+01   -0.490000000000000000E+01   -0.590000000000000000E+01   -0.690000000000000000E+01   -0.790000000000000000E+01    0.890000000000000000E+01    0.990000000000000000E+01    1.090000000000000000E+01\n'
                           '    0.010000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01\n'
                           '    0.050000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01\n'
                           '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01\n'
                           '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01\n'
                           '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01\n'
                           '    0.100000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01\n'
                           '    0.200000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01\n'
                           '    0.300000000000000000E+00    0.999999990000000000E+07    0.100000000000000000E+01    0.200000000000000000E+01    0.300000000000000000E+01    0.400000000000000000E+01   -0.500000000000000000E+01   -0.600000000000000000E+01   -0.700000000000000000E+01   -0.800000000000000000E+01    0.900000000000000000E+01    1.000000000000000000E+01    1.100000000000000000E+01')
    weighted_samples.close()

class TestNonLinearFiles(object):

    class TestDirectorySetup(object):

        def test__one_light_profile__correct_directory(self):

            if os.path.exists(path + 'test_files/non_linear/files/setup/obj/EllipticalSersic'):
                shutil.rmtree(path + 'test_files/non_linear/files/setup/obj/EllipticalSersic')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic)

            non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/setup/',
                                      obj_name='obj', model_mapper=model_map)

            assert os.path.exists(path + 'test_files/non_linear/files/setup/obj/EllipticalSersic') == True

        def test__one_mass_profile__correct_directory(self):

            if os.path.exists(path + 'test_files/non_linear/files/setup/obj/SphericalNFW'):
                shutil.rmtree(path + 'test_files/non_linear/files/setup/obj/SphericalNFW')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/setup/',
                                      obj_name='obj', model_mapper=model_map)

            assert os.path.exists(path + 'test_files/non_linear/files/setup/obj/SphericalNFW') == True

        def test__multiple_light_and_mass_profiles__correct_directory(self):

            if os.path.exists(path +
                              'test_files/non_linear/files/setup/obj/'
                              'EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW'):
                shutil.rmtree(
                    path + 'test_files/non_linear/files/setup/obj/'
                           'EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic,
                                                 light_profile_2=light_profiles.EllipticalSersic,
                                                 light_profile_3=light_profiles.EllipticalSersic,
                                                 mass_profile=mass_profiles.SphericalNFW,
                                                 mass_profile_2=mass_profiles.SphericalNFW)

            non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/setup/',
                                      obj_name='obj', model_mapper=model_map)

            assert os.path.exists(path +
                                  'test_files/non_linear/files/setup/obj/'
                                  'EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW') == True

    class TestTotalParameters(object):

        def test__one_light_profile__correct_directory(self):

            if os.path.exists(path + 'test_files/non_linear/files/obj/EllipticalSersic'):
                shutil.rmtree(path + 'test_files/non_linear/files/obj/EllipticalSersic')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic)

            nl_directory = non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/',
                                                     obj_name='obj', model_mapper=model_map)

            assert nl_directory.total_parameters == 7

        def test__one_mass_profile__correct_directory(self):

            if os.path.exists(path + 'test_files/non_linear/files/obj/SphericalNFW'):
                shutil.rmtree(path + 'test_files/non_linear/files/obj/SphericalNFW')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            nl_directory = non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/',
                                                     obj_name='obj', model_mapper=model_map)

            assert nl_directory.total_parameters == 4

        def test__nl_directory_multiple_light_and_mass_profiles__correct_directory(self):

            if os.path.exists(path +
                              'test_files/non_linear/files/obj/'
                              'EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW'):
                shutil.rmtree(
                    path + 'test_files/non_linear/files/obj/'
                           'EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic,
                                                 light_profile_2=light_profiles.EllipticalSersic,
                                                 light_profile_3=light_profiles.EllipticalSersic,
                                                 mass_profile=mass_profiles.SphericalNFW,
                                                 mass_profile_2=mass_profiles.SphericalNFW)

            nl_directory = non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/',
                                                     obj_name='obj', model_mapper=model_map)

            assert nl_directory.total_parameters == 29

    class TestGenerateLatex(object):

        def test__one_parameter__no_subscript(self):
            assert non_linear.generate_parameter_latex('x') == ['$x$']

        def test__three_parameters__no_subscript(self):
            assert non_linear.generate_parameter_latex(['x', 'y', 'z']) == ['$x$', '$y$', '$z$']

        def test__one_parameter__subscript__no_number(self):
            assert non_linear.generate_parameter_latex(['x'], subscript='d') == [r'$x_{\mathrm{d}}$']

        def test__three_parameters__subscript__no_number(self):
            assert non_linear.generate_parameter_latex(['x', 'y', 'z'], subscript='d') == [r'$x_{\mathrm{d}}$',
                                                                                           r'$y_{\mathrm{d}}$',
                                                                                           r'$z_{\mathrm{d}}$']

    class TestCreateParamNames(object):

        def test__single_model_and_parameter_set__outputs_paramnames(self):

            light_profiles.EllipticalLightProfile._ids = count()

            if os.path.exists(path + 'test_files/non_linear/files/param_names/obj/EllipticalSersic'):
                shutil.rmtree(path + 'test_files/non_linear/files/param_names/obj/EllipticalSersic')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic)

            non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/param_names/',
                                      obj_name='obj', model_mapper=model_map)

            paramnames_test = open(path + 'test_files/non_linear/files/param_names/obj/'
                                          'EllipticalSersic/obj.paramnames')

            paramnames_str_0 = paramnames_test.readline()
            paramnames_str_1 = paramnames_test.readline()
            paramnames_str_2 = paramnames_test.readline()
            paramnames_str_3 = paramnames_test.readline()
            paramnames_str_4 = paramnames_test.readline()
            paramnames_str_5 = paramnames_test.readline()
            paramnames_str_6 = paramnames_test.readline()

            assert paramnames_str_0 == r'light_profile_0_centre_0                $x_{\mathrm{l1}}$' + '\n'
            assert paramnames_str_1 == r'light_profile_0_centre_1                $y_{\mathrm{l1}}$' + '\n'
            assert paramnames_str_2 == r'light_profile_0_axis_ratio              $q_{\mathrm{l1}}$' + '\n'
            assert paramnames_str_3 == r'light_profile_0_phi                     $\phi_{\mathrm{l1}}$' + '\n'
            assert paramnames_str_4 == r'light_profile_0_intensity               $I_{\mathrm{l1}}$' + '\n'
            assert paramnames_str_5 == r'light_profile_0_effective_radius        $R_{\mathrm{l1}}$' + '\n'
            assert paramnames_str_6 == r'light_profile_0_sersic_index            $n_{\mathrm{l1}}$' + '\n'

        def test__two_light_models_outputs_paramnames(self):

            light_profiles.EllipticalLightProfile._ids = count()

            if os.path.exists(path + 'test_files/non_linear/files/param_names/obj/'
                                     'EllipticalSersic+EllipticalExponential'):
                shutil.rmtree(path + 'test_files/non_linear/files/param_names/obj/'
                                     'EllipticalSersic+EllipticalExponential')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic,
                                                 light_profile_1=light_profiles.EllipticalExponential)

            non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/param_names/',
                                          obj_name='obj', model_mapper=model_map)

            paramnames_test = open(path + 'test_files/non_linear/files/param_names/obj/'
                                          'EllipticalSersic+EllipticalExponential/obj.paramnames')

            paramnames_str = paramnames_test.readlines()

            assert paramnames_str[0] == r'light_profile_0_centre_0                $x_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[1] == r'light_profile_0_centre_1                $y_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[2] == r'light_profile_0_axis_ratio              $q_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[3] == r'light_profile_0_phi                     $\phi_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[4] == r'light_profile_0_intensity               $I_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[5] == r'light_profile_0_effective_radius        $R_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[6] == r'light_profile_0_sersic_index            $n_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[7] == r'light_profile_1_centre_0                $x_{\mathrm{l2}}$' + '\n'
            assert paramnames_str[8] == r'light_profile_1_centre_1                $y_{\mathrm{l2}}$' + '\n'
            assert paramnames_str[9] == r'light_profile_1_axis_ratio              $q_{\mathrm{l2}}$' + '\n'
            assert paramnames_str[10] == r'light_profile_1_phi                     $\phi_{\mathrm{l2}}$' + '\n'
            assert paramnames_str[11] == r'light_profile_1_intensity               $I_{\mathrm{l2}}$' + '\n'
            assert paramnames_str[12] == r'light_profile_1_effective_radius        $R_{\mathrm{l2}}$' + '\n'

        def test__two_light_models__two_mass_models__outputs_paramnames(self):

            light_profiles.EllipticalLightProfile._ids = count()
            mass_profiles.EllipticalMassProfile._ids = count()

            if os.path.exists(path + 'test_files/non_linear/files/param_names/obj/'
                                     'EllipticalSersic+EllipticalExponential+SphericalIsothermal+SphericalNFW'):
                shutil.rmtree(path + 'test_files/non_linear/files/param_names/obj/'
                                     'EllipticalSersic+EllipticalExponential+SphericalIsothermal+SphericalNFW')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic,
                                                 light_profile_1=light_profiles.EllipticalExponential,
                                                 mass_profile_0=mass_profiles.SphericalIsothermal,
                                                 mass_profile_1=mass_profiles.SphericalNFW)

            non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/param_names/',
                                          obj_name='obj', model_mapper=model_map)

            paramnames_test = open(path + 'test_files/non_linear/files/param_names/obj/'
                                          'EllipticalSersic+EllipticalExponential+SphericalIsothermal+SphericalNFW/obj.paramnames')

            paramnames_str = paramnames_test.readlines()

            assert paramnames_str[0] == r'light_profile_0_centre_0                $x_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[1] == r'light_profile_0_centre_1                $y_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[2] == r'light_profile_0_axis_ratio              $q_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[3] == r'light_profile_0_phi                     $\phi_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[4] == r'light_profile_0_intensity               $I_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[5] == r'light_profile_0_effective_radius        $R_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[6] == r'light_profile_0_sersic_index            $n_{\mathrm{l1}}$' + '\n'
            assert paramnames_str[7] == r'light_profile_1_centre_0                $x_{\mathrm{l2}}$' + '\n'
            assert paramnames_str[8] == r'light_profile_1_centre_1                $y_{\mathrm{l2}}$' + '\n'
            assert paramnames_str[9] == r'light_profile_1_axis_ratio              $q_{\mathrm{l2}}$' + '\n'
            assert paramnames_str[10] == r'light_profile_1_phi                     $\phi_{\mathrm{l2}}$' + '\n'
            assert paramnames_str[11] == r'light_profile_1_intensity               $I_{\mathrm{l2}}$' + '\n'
            assert paramnames_str[12] == r'light_profile_1_effective_radius        $R_{\mathrm{l2}}$' + '\n'
            assert paramnames_str[13] == r'mass_profile_0_centre_0                 $x_{\mathrm{1}}$' + '\n'
            assert paramnames_str[14] == r'mass_profile_0_centre_1                 $y_{\mathrm{1}}$' + '\n'
            assert paramnames_str[15] == r'mass_profile_0_einstein_radius          $\theta_{\mathrm{1}}$' + '\n'
            assert paramnames_str[16] == r'mass_profile_1_centre_0                 $x_{\mathrm{d2}}$' + '\n'
            assert paramnames_str[17] == r'mass_profile_1_centre_1                 $y_{\mathrm{d2}}$' + '\n'
            assert paramnames_str[18] == r'mass_profile_1_kappa_s                  $\kappa_{\mathrm{d2}}$' + '\n'
            assert paramnames_str[19] == r'mass_profile_1_scale_radius             $Rs_{\mathrm{d2}}$' + '\n'

    class TestMakeModelInfo(object):

        def test__single_model__outputs_all_info(self):

            if os.path.exists(path + 'test_files/non_linear/files/model_info/obj/EllipticalSersic'):
                shutil.rmtree(path + 'test_files/non_linear/files/model_info/obj/EllipticalSersic')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic)

            non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/model_info/',
                                      obj_name='obj', model_mapper=model_map)

            model_info_test = open(path + 'test_files/non_linear/files/model_info/obj/'
                                          'EllipticalSersic/model.info')

            model_info_str = model_info_test.readlines()

            assert model_info_str[0] == r'EllipticalSersic' + '\n'
            assert model_info_str[1] == r'' + '\n'
            assert model_info_str[2] == r'centre_0: UniformPrior, lower_limit = 0.0, upper_limit = 1.0' + '\n'
            assert model_info_str[3] == r'centre_1: UniformPrior, lower_limit = 0.0, upper_limit = 0.5' + '\n'
            assert model_info_str[4] == r'axis_ratio: UniformPrior, lower_limit = 0.0, upper_limit = 0.5' + '\n'
            assert model_info_str[5] == r'phi: UniformPrior, lower_limit = 0.0, upper_limit = 0.5' + '\n'
            assert model_info_str[6] == r'intensity: GaussianPrior, mean = 0.0, sigma = 0.5' + '\n'
            assert model_info_str[7] == r'effective_radius: UniformPrior, lower_limit = 1.0, upper_limit = 1.0' + '\n'
            assert model_info_str[8] == r'sersic_index: UniformPrior, lower_limit = 1.0, upper_limit = 1.0' + '\n'

        def test__two_models_and_parameter_sets__outputs_paramnames(self):

            if os.path.exists(path + 'test_files/non_linear/files/model_info/obj/'
                                     'EllipticalSersic+EllipticalExponential+SphericalIsothermal+SphericalNFW'):
                shutil.rmtree(path + 'test_files/non_linear/files/model_info/obj/'
                                     'EllipticalSersic+EllipticalExponential+SphericalIsothermal+SphericalNFW')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic,
                                                 light_profile_1=light_profiles.EllipticalExponential,
                                                 mass_profile_0=mass_profiles.SphericalIsothermal,
                                                 mass_profile_1=mass_profiles.SphericalNFW)

            non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/model_info/',
                                          obj_name='obj', model_mapper=model_map)

            model_info_test = open(
                path + 'test_files/non_linear/files/model_info/obj/'
                       'EllipticalSersic+EllipticalExponential+SphericalIsothermal+SphericalNFW/model.info')

            model_info_str = model_info_test.readlines()

            assert model_info_str[0] == r'EllipticalSersic' + '\n'
            assert model_info_str[1] == r'' + '\n'
            assert model_info_str[2] == r'centre_0: UniformPrior, lower_limit = 0.0, upper_limit = 1.0' + '\n'
            assert model_info_str[3] == r'centre_1: UniformPrior, lower_limit = 0.0, upper_limit = 0.5' + '\n'
            assert model_info_str[4] == r'axis_ratio: UniformPrior, lower_limit = 0.0, upper_limit = 0.5' + '\n'
            assert model_info_str[5] == r'phi: UniformPrior, lower_limit = 0.0, upper_limit = 0.5' + '\n'
            assert model_info_str[6] == r'intensity: GaussianPrior, mean = 0.0, sigma = 0.5' + '\n'
            assert model_info_str[7] == r'effective_radius: UniformPrior, lower_limit = 1.0, upper_limit = 1.0' + '\n'
            assert model_info_str[8] == r'sersic_index: UniformPrior, lower_limit = 1.0, upper_limit = 1.0' + '\n'
            assert model_info_str[9] == r'' + '\n'
            assert model_info_str[10] == r'EllipticalExponential' + '\n'
            assert model_info_str[11] == r'' + '\n'
            assert model_info_str[12] == r'centre_0: UniformPrior, lower_limit = 0.0, upper_limit = 1.0' + '\n'
            assert model_info_str[13] == r'centre_1: UniformPrior, lower_limit = 0.0, upper_limit = 0.5' + '\n'
            assert model_info_str[14] == r'axis_ratio: UniformPrior, lower_limit = 0.0, upper_limit = 0.5' + '\n'
            assert model_info_str[15] == r'phi: UniformPrior, lower_limit = 0.0, upper_limit = 0.5' + '\n'
            assert model_info_str[16] == r'intensity: GaussianPrior, mean = 0.0, sigma = 0.5' + '\n'
            assert model_info_str[17] == r'effective_radius: UniformPrior, lower_limit = 1.0, upper_limit = 1.0' + '\n'
            assert model_info_str[18] == r'' + '\n'
            assert model_info_str[19] == r'SphericalIsothermal' + '\n'
            assert model_info_str[20] == r'' + '\n'
            assert model_info_str[21] == r'centre_0: UniformPrior, lower_limit = 0.0, upper_limit = 1.0' + '\n'
            assert model_info_str[22] == r'centre_1: UniformPrior, lower_limit = 0.0, upper_limit = 0.5' + '\n'
            assert model_info_str[23] == r'einstein_radius: UniformPrior, lower_limit = 1.0, upper_limit = 1.0' + '\n'
            assert model_info_str[24] == r'' + '\n'
            assert model_info_str[25] == r'SphericalNFW' + '\n'
            assert model_info_str[26] == r'' + '\n'
            assert model_info_str[27] == r'centre_0: UniformPrior, lower_limit = 0.0, upper_limit = 1.0' + '\n'
            assert model_info_str[28] == r'centre_1: UniformPrior, lower_limit = 0.0, upper_limit = 0.5' + '\n'
            assert model_info_str[29] == r'kappa_s: UniformPrior, lower_limit = 1.0, upper_limit = 1.0' + '\n'
            assert model_info_str[30] == r'scale_radius: UniformPrior, lower_limit = 1.0, upper_limit = 1.0' + '\n'

    class TestCheckModelInfo(object):

        def test__single_model__prior_changed_from_input_model__raises_error(self):

            if os.path.exists(path + 'test_files/non_linear/files/wrong_model_info/obj/EllipticalSersic/*'):
                shutil.rmtree(path + 'test_files/non_linear/files/wrong_model_info/obj/EllipticalSersic/*')

            with open(path+'test_files/non_linear/files/wrong_model_info/obj/EllipticalSersic/model.info', 'w') as file:
                file.write('The model info is missing :(')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic)

            with pytest.raises(model_mapper.PriorException):
                non_linear.NonLinearFiles(path=path + 'test_files/non_linear/files/wrong_model_info/',
                                          obj_name='obj', model_mapper=model_map)


class TestMultiNest(object):

    class TestReadFromSummary:

        def test__1_profile__read_most_likely_vector__via_summary(self):

            if os.path.exists(path+'test_files/non_linear/multinest/summary/obj/SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinest/summary/obj/SphericalNFW/*')

            create_summary_4_parameters(path=path+'test_files/non_linear/multinest/summary/obj/SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            files = non_linear.MultiNest(path=path + 'test_files/non_linear/multinest/summary/',
                                                  obj_name='obj', model_mapper=model_map, check_model=False)

            most_likely = files.compute_most_likely()

            assert most_likely == [-5.0, 6.0, 7.0, 8.0]

        def test__multiple_profile__read_most_likely_vector__via_summary(self):

            if os.path.exists(path+'test_files/non_linear/multinest/summary/obj/EllipticalSersic+SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinest/summary/obj/EllipticalSersic+SphericalNFW/*')

            create_summary_11_parameters(path=path+'test_files/non_linear/multinest/summary/obj/'
                                                  'EllipticalSersic+SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                 light_profile=light_profiles.EllipticalSersic,
                                                 mass_profile=mass_profiles.SphericalNFW)

            files = non_linear.MultiNest(path=path + 'test_files/non_linear/multinest/summary/',
                                                  obj_name='obj', model_mapper=model_map, check_model=False)

            most_likely = files.compute_most_likely()

            assert most_likely == [12.0, 13.0, 14.0, 15.0, 16.0, -17.0, -18.0, -19.0, -20.0, 21.0, 22.0]

        def test__1_profile__read_most_probable_vector__via_summary(self):

            if os.path.exists(path+'test_files/non_linear/multinest/summary/obj/SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinest/summary/obj/SphericalNFW/*')

            create_summary_4_parameters(path=path+'test_files/non_linear/multinest/summary/obj/SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            files = non_linear.MultiNest(path=path + 'test_files/non_linear/multinest/summary/',
                                                  obj_name='obj', model_mapper=model_map, check_model=False)

            most_probable = files.compute_most_probable()

            assert most_probable == [1.0, -2.0, 3.0, 4.0]

        def test__multiple_profile__read_most_probable_vector__via_summary(self):

            if os.path.exists(path+'test_files/non_linear/multinest/summary/obj/EllipticalSersic+SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinest/summary/obj/EllipticalSersic+SphericalNFW/*')

            create_summary_11_parameters(path=path+'test_files/non_linear/multinest/summary/obj/'
                                                  'EllipticalSersic+SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                 light_profile=light_profiles.EllipticalSersic,
                                                 mass_profile=mass_profiles.SphericalNFW)

            files = non_linear.MultiNest(path=path + 'test_files/non_linear/multinest/summary/',
                                                  obj_name='obj', model_mapper=model_map, check_model=False)

            most_probable = files.compute_most_probable()

            assert most_probable == [1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0, 11.0]

        def test__1_profile__read_likelihoods_from_summary(self):

            if os.path.exists(path+'test_files/non_linear/multinest/summary/obj/SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinest/summary/obj/SphericalNFW/*')

            create_summary_4_parameters(path=path+'test_files/non_linear/multinest/summary/obj/SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            files = non_linear.MultiNest(path=path + 'test_files/non_linear/multinest/summary/',
                                                  obj_name='obj', model_mapper=model_map, check_model=False)

            max_likelihood = files.compute_max_likelihood()
            max_log_likelihood = files.compute_max_log_likelihood()

            assert max_likelihood == 0.02
            assert max_log_likelihood == 9999999.9

        def test__multiple_profiles__read_likelihoods_from_summary(self):

            if os.path.exists(path+'test_files/non_linear/multinest/summary/obj/EllipticalSersic+SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinest/summary/obj/EllipticalSersic+SphericalNFW/*')

            create_summary_11_parameters(path=path+'test_files/non_linear/multinest/summary/obj/'
                                                  'EllipticalSersic+SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                 light_profile=light_profiles.EllipticalSersic,
                                                 mass_profile=mass_profiles.SphericalNFW)

            files = non_linear.MultiNest(path=path + 'test_files/non_linear/multinest/summary/',
                                                  obj_name='obj', model_mapper=model_map, check_model=False)

            max_likelihood = files.compute_max_likelihood()
            max_log_likelihood = files.compute_max_log_likelihood()

            assert max_likelihood == 0.02
            assert max_log_likelihood == 9999999.9

        def test__multiple_profiles__setup_model_instance_most_likely_and_probable_via_summary(self):

            if os.path.exists(
                    path + 'test_files/non_linear/multinest/summary/obj/EllipticalSersic+SphericalNFW/*'):
                shutil.rmtree(
                    path + 'test_files/non_linear/multinest/results_intermediate/summary/obj/EllipticalSersic+SphericalNFW/*')

            create_summary_11_parameters(path=path + 'test_files/non_linear/multinest/summary/obj/'
                                                     'EllipticalSersic+SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(
                config=config, light_profile=light_profiles.EllipticalSersic, mass_profile=mass_profiles.SphericalNFW)

            multinest = non_linear.MultiNest(path=path + 'test_files/non_linear/multinest/summary/',
                obj_name='obj', model_mapper=model_map, check_model=False)

            most_probable = multinest.create_most_probable_model_instance()
            most_likely = multinest.create_most_likely_model_instance()

            assert most_probable.light_profile.centre == (1.0, 2.0)
            assert most_probable.light_profile.axis_ratio == 3.0
            assert most_probable.light_profile.phi == 4.0
            assert most_probable.light_profile.intensity == -5.0
            assert most_probable.light_profile.effective_radius == -6.0
            assert most_probable.light_profile.sersic_index == -7.0

            assert most_probable.mass_profile.centre == (-8.0, 9.0)
            assert most_probable.mass_profile.kappa_s == 10.0
            assert most_probable.mass_profile.scale_radius == 11.0

            assert most_likely.light_profile.centre == (12.0, 13.0)
            assert most_likely.light_profile.axis_ratio == 14.0
            assert most_likely.light_profile.phi == 15.0
            assert most_likely.light_profile.intensity == 16.0
            assert most_likely.light_profile.effective_radius == -17.0
            assert most_likely.light_profile.sersic_index == -18.0

            assert most_likely.mass_profile.centre == (-19.0, -20.0)
            assert most_likely.mass_profile.kappa_s == 21.0
            assert most_likely.mass_profile.scale_radius == 22.0


class TestMultiNestFinished(object):

    class TestWeightedSamples(object):

        def test__one_profile__read_first_weighted_sample__model_weight_and_likelihood(self):

            if os.path.exists(path+'test_files/non_linear/multinestfin/weighted_samples/obj/SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinestfin/weighted_samples/obj/SphericalNFW/*')

            create_weighted_samples_4_parameters(path=path+'test_files/non_linear/multinestfin/weighted_samples/obj/'
                                                           'SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                 mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestFinished(path=path + 'test_files/non_linear/multinestfin/weighted_samples/',
                                                   obj_name='obj', model_mapper=model_map, check_model=False)

            model, weight, likelihood = results.compute_weighted_sample_model(index=0)

            assert model == [1.1, 2.1, 3.1, 4.1]
            assert weight == 0.02
            assert likelihood == -0.5 * 9999999.9

        def test__one_profile__read_fifth_weighted_sample__model_weight_and_likelihood(self):

            if os.path.exists(path+'test_files/non_linear/multinestfin/weighted_samples/obj/SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinestfin/weighted_samples/obj/SphericalNFW/*')

            create_weighted_samples_4_parameters(path=path+'test_files/non_linear/multinestfin/weighted_samples/obj/'
                                                           'SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                 mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestFinished(path=path + 'test_files/non_linear/multinestfin/weighted_samples/',
                                                   obj_name='obj', model_mapper=model_map, check_model=False)

            model, weight, likelihood = results.compute_weighted_sample_model(index=5)

            assert model == [1.0, 2.0, 3.0, 4.0]
            assert weight == 0.1
            assert likelihood == -0.5 * 9999999.9

        def test__multiple_profiles__read_first_weighted_sample__model_weight_and_likelihood(self):

            if os.path.exists(path+'test_files/non_linear/multinestfin/weighted_samples/obj/EllipticalSersic+SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinestfin/weighted_samples/obj/EllipticalSersic+SphericalNFW/*')

            create_weighted_samples_11_parameters(path=path+'test_files/non_linear/multinestfin/weighted_samples/obj/'
                                                           'EllipticalSersic+SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                 light_profile=light_profiles.EllipticalSersic,
                                                 mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestFinished(path=path + 'test_files/non_linear/multinestfin/weighted_samples/',
                                                   obj_name='obj', model_mapper=model_map, check_model=False)

            model, weight, likelihood = results.compute_weighted_sample_model(index=0)

            assert model == [1.1, 2.1, 3.1, 4.1, -5.1, -6.1, -7.1, -8.1, 9.1, 10.1, 11.1]
            assert weight == 0.02
            assert likelihood == -0.5 * 9999999.9

        def test__multiple_profiles__read_fifth_weighted_sample__model_weight_and_likelihood(self):

            if os.path.exists(path+'test_files/non_linear/multinestfin/weighted_samples/obj/EllipticalSersic+SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinestfin/weighted_samples/obj/EllipticalSersic+SphericalNFW/*')

            create_weighted_samples_11_parameters(path=path+'test_files/non_linear/multinestfin/weighted_samples/obj/'
                                                           'EllipticalSersic+SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                 light_profile=light_profiles.EllipticalSersic,
                                                 mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestFinished(path=path + 'test_files/non_linear/multinestfin/weighted_samples/',
                                                   obj_name='obj', model_mapper=model_map, check_model=False)

            model, weight, likelihood = results.compute_weighted_sample_model(index=5)

            assert model == [1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0, 11.0]
            assert weight == 0.1
            assert likelihood == -0.5 * 9999999.9

        def test__one_profile__setup_first_weighted_sample_model__include_weight_and_likelihood(self):

            if os.path.exists(path+'test_files/non_linear/multinestfin/weighted_samples/obj/EllipticalSersic+SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinestfin/weighted_samples/obj/EllipticalSersic+SphericalNFW/*')

            create_weighted_samples_11_parameters(path=path+'test_files/non_linear/multinestfin/weighted_samples/obj/'
                                                           'EllipticalSersic+SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                 mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestFinished(path=path + 'test_files/non_linear/multinestfin/weighted_samples/',
                                                   obj_name='obj', model_mapper=model_map, check_model=False)

            weighted_sample_model, weight, likelihood = results.create_weighted_sample_model_instance(index=0)

            assert weight == 0.02
            assert likelihood == -0.5 * 9999999.9

            assert weighted_sample_model.mass_profile.centre == (1.1, 2.1)
            assert weighted_sample_model.mass_profile.kappa_s == 3.1
            assert weighted_sample_model.mass_profile.scale_radius == 4.1

        def test__1_profile__setup_fifth_weighted_sample_model__include_weight_and_likelihood(self):

            if os.path.exists(path+'test_files/non_linear/multinestfin/weighted_samples/obj/SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinestfin/weighted_samples/obj/SphericalNFW/*')

            create_weighted_samples_4_parameters(path=path+'test_files/non_linear/multinestfin/weighted_samples/obj/'
                                                           'SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                 mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestFinished(path=path + 'test_files/non_linear/multinestfin/weighted_samples/',
                                                   obj_name='obj', model_mapper=model_map, check_model=False)

            weighted_sample_model, weight, likelihood = results.create_weighted_sample_model_instance(index=5)

            assert weight == 0.1
            assert likelihood == -0.5 * 9999999.9

            assert weighted_sample_model.mass_profile.centre == (1.0, 2.0)
            assert weighted_sample_model.mass_profile.kappa_s == 3.0
            assert weighted_sample_model.mass_profile.scale_radius == 4.0

        def test__multiple_profiles__setup_first_weighted_sample_model__include_weight_and_likelihood(self):

            if os.path.exists(path+'test_files/non_linear/multinestfin/weighted_samples/obj/EllipticalSersic+SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinestfin/weighted_samples/obj/EllipticalSersic+SphericalNFW/*')

            create_weighted_samples_11_parameters(path=path+'test_files/non_linear/multinestfin/weighted_samples/obj/'
                                                           'EllipticalSersic+SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                 light_profile=light_profiles.EllipticalSersic,
                                                 mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestFinished(path=path + 'test_files/non_linear/multinestfin/weighted_samples/',
                                                   obj_name='obj', model_mapper=model_map, check_model=False)

            weighted_sample_model, weight, likelihood = results.create_weighted_sample_model_instance(index=0)

            assert weight == 0.02
            assert likelihood == -0.5 * 9999999.9

            assert weighted_sample_model.light_profile.centre == (1.1, 2.1)
            assert weighted_sample_model.light_profile.axis_ratio == 3.1
            assert weighted_sample_model.light_profile.phi == 4.1
            assert weighted_sample_model.light_profile.intensity == -5.1
            assert weighted_sample_model.light_profile.effective_radius == -6.1
            assert weighted_sample_model.light_profile.sersic_index == -7.1

            assert weighted_sample_model.mass_profile.centre == (-8.1, 9.1)
            assert weighted_sample_model.mass_profile.kappa_s == 10.1
            assert weighted_sample_model.mass_profile.scale_radius == 11.1

        def test__multiple_profiles__setup_fifth_weighted_sample_model__include_weight_and_likelihood(self):

            if os.path.exists(path+'test_files/non_linear/multinestfin/weighted_samples/obj/EllipticalSersic+SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinestfin/weighted_samples/obj/EllipticalSersic+SphericalNFW/*')

            create_weighted_samples_11_parameters(path=path+'test_files/non_linear/multinestfin/weighted_samples/obj/'
                                                           'EllipticalSersic+SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                 light_profile=light_profiles.EllipticalSersic,
                                                 mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestFinished(path=path + 'test_files/non_linear/multinestfin/weighted_samples/',
                                                   obj_name='obj', model_mapper=model_map, check_model=False)

            weighted_sample_model, weight, likelihood = results.create_weighted_sample_model_instance(index=5)

            assert weight == 0.1
            assert likelihood == -0.5 * 9999999.9

            assert weighted_sample_model.light_profile.centre == (1.0, 2.0)
            assert weighted_sample_model.light_profile.axis_ratio == 3.0
            assert weighted_sample_model.light_profile.phi == 4.0
            assert weighted_sample_model.light_profile.intensity == -5.0
            assert weighted_sample_model.light_profile.effective_radius == -6.0
            assert weighted_sample_model.light_profile.sersic_index == -7.0

            assert weighted_sample_model.mass_profile.centre == (-8.0, 9.0)
            assert weighted_sample_model.mass_profile.kappa_s == 10.0
            assert weighted_sample_model.mass_profile.scale_radius == 11.0

    class TestLimits(object):

        def test__1_profile__limits_1d_vectors_via_weighted_samples__1d_vectors_are_correct(self):

            if os.path.exists(path+'test_files/non_linear/multinestfin/weighted_samples/obj/SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinestfin/weighted_samples/obj/SphericalNFW/*')

            create_weighted_samples_4_parameters(path=path+'test_files/non_linear/multinestfin/weighted_samples/obj/'
                                                           'SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestFinished(path=path + 'test_files/non_linear/multinestfin/weighted_samples/',
                                                   obj_name='obj', model_mapper=model_map, check_model=False)

            assert results.compute_model_at_upper_limit(limit=0.9973) == pytest.approx([1.12, 2.12, 3.12, 4.12], 1e-2)
            assert results.compute_model_at_lower_limit(limit=0.9973) == pytest.approx([0.88, 1.88, 2.88, 3.88], 1e-2)

        def test__1_profile__change_limit_to_1_sigma(self):

            if os.path.exists(path+'test_files/non_linear/multinestfin/weighted_samples/obj/SphericalNFW/*'):
                shutil.rmtree(path+'test_files/non_linear/multinestfin/weighted_samples/obj/SphericalNFW/*')

            create_weighted_samples_4_parameters(path=path+'test_files/non_linear/multinestfin/weighted_samples/obj/'
                                                           'SphericalNFW/')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestFinished(path=path + 'test_files/non_linear/multinestfin/weighted_samples/',
                                                   obj_name='obj', model_mapper=model_map, check_model=False)

            assert results.compute_model_at_upper_limit(limit=0.6827) == pytest.approx([1.07, 2.07, 3.07, 4.07], 1e-2)
            assert results.compute_model_at_lower_limit(limit=0.6827) == pytest.approx([0.93, 1.93, 2.93, 3.93], 1e-2)