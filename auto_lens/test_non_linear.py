import os
import shutil
import pytest
from itertools import count
from auto_lens import non_linear
from auto_lens import model_mapper
from auto_lens.profiles import geometry_profiles, light_profiles, mass_profiles

path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))


class TestNonLinearDirectory(object):


    class TestDirectorySetup(object):
    
        def test__one_light_profile__correct_directory(self):
    
            if os.path.exists(path + 'test_files/non_linear/directory/setup/EllipticalSersic'):
                shutil.rmtree(path + 'test_files/non_linear/directory/setup/EllipticalSersic')
    
            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic)
    
            non_linear.NonLinearDirectory(path=path + 'test_files/non_linear/directory/setup/', model_mapper=model_map)
    
            assert os.path.exists(path + 'test_files/non_linear/directory/setup/EllipticalSersic') == True
    
        def test__one_mass_profile__correct_directory(self):
            
            if os.path.exists(path + 'test_files/non_linear/directory/setup/SphericalNFW'):
                shutil.rmtree(path + 'test_files/non_linear/directory/setup/SphericalNFW')
    
            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)
    
            non_linear.NonLinearDirectory(path=path + 'test_files/non_linear/directory/setup/', model_mapper=model_map)
    
            assert os.path.exists(path + 'test_files/non_linear/directory/setup/SphericalNFW') == True
    
        def test__multiple_light_and_mass_profiles__correct_directory(self):
            
            if os.path.exists(path +
                    'test_files/non_linear/directory/setup/EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW'):
                shutil.rmtree(
                    path + 'test_files/non_linear/directory/setup/EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW')
    
            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic,
                                                  light_profile_2=light_profiles.EllipticalSersic,
                                                  light_profile_3=light_profiles.EllipticalSersic,
                                                  mass_profile=mass_profiles.SphericalNFW,
                                                  mass_profile_2=mass_profiles.SphericalNFW)
    
            non_linear.NonLinearDirectory(path=path + 'test_files/non_linear/directory/setup/', model_mapper=model_map)
    
            assert os.path.exists(path +
                    'test_files/non_linear/directory/setup/EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW') == True


    class TestTotalParameters(object):
    
        def test__one_light_profile__correct_directory(self):
    
            if os.path.exists(path + 'test_files/non_linear/directory_setup/EllipticalSersic'):
                shutil.rmtree(path + 'test_files/non_linear/directory_setup/EllipticalSersic')
    
            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic)
    
            nl_directory = non_linear.NonLinearDirectory(path=path + 'test_files/non_linear/directory_setup/', model_mapper=model_map)
    
            assert nl_directory.total_parameters == 7
    
        def test__one_mass_profile__correct_directory(self):
    
            if os.path.exists(path + 'test_files/non_linear/directory_setup/SphericalNFW'):
                shutil.rmtree(path + 'test_files/non_linear/directory_setup/SphericalNFW')
    
            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)
    
            nl_directory = non_linear.NonLinearDirectory(path=path + 'test_files/non_linear/directory_setup/', model_mapper=model_map)
    
            assert nl_directory.total_parameters == 4
    
        def test__nl_directoryple_light_and_mass_profiles__correct_directory(self):
    
            if os.path.exists(path +
                              'test_files/non_linear/directory_setup/EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW'):
                shutil.rmtree(
                    path + 'test_files/non_linear/directory_setup/EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW')
    
            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic,
                                                  light_profile_2=light_profiles.EllipticalSersic,
                                                  light_profile_3=light_profiles.EllipticalSersic,
                                                  mass_profile=mass_profiles.SphericalNFW,
                                                  mass_profile_2=mass_profiles.SphericalNFW)
    
            nl_directory = non_linear.NonLinearDirectory(path=path + 'test_files/non_linear/directory_setup/', model_mapper=model_map)
    
            assert nl_directory.total_parameters == 29


    class TestGenerateLatex(object):

        def test__one_parameter__no_subscript(self):
            
            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic)

            nl_directory = non_linear.NonLinearDirectory(path=path + 'test_files/non_linear/directory_setup/',
                                                         model_mapper=model_map)

            assert nl_directory.generate_parameter_latex('x') == ['$x$']

        def test__three_parameters__no_subscript(self):
            
            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic)

            nl_directory = non_linear.NonLinearDirectory(path=path + 'test_files/non_linear/directory_setup/',
                                                         model_mapper=model_map)

            assert nl_directory.generate_parameter_latex(['x', 'y', 'z']) == ['$x$', '$y$', '$z$']

        def test__one_parameter__subscript__no_number(self):
            
            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic)

            nl_directory = non_linear.NonLinearDirectory(path=path + 'test_files/non_linear/directory_setup/',
                                                         model_mapper=model_map)

            assert nl_directory.generate_parameter_latex(['x'], subscript='d') == [r'$x_{\mathrm{d}}$']

        def test__three_parameters__subscript__no_number(self):
            
            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic)

            nl_directory = non_linear.NonLinearDirectory(path=path + 'test_files/non_linear/directory_setup/',
                                                         model_mapper=model_map)

            assert nl_directory.generate_parameter_latex(['x', 'y', 'z'], subscript='d') == [r'$x_{\mathrm{d}}$',
                                                                                           r'$y_{\mathrm{d}}$',
                                                                                           r'$z_{\mathrm{d}}$']


class TestMultiNest(object):


    class TestInheritance(object):

        def test__directory_parameters_and_latex_all_work(self):

            if os.path.exists(path + 'test_files/non_linear/multinest/optimizer/directory_setup/EllipticalSersic'):
                shutil.rmtree(path + 'test_files/non_linear/multinest/optimizer/directory_setup/EllipticalSersic')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic)

            multi = non_linear.MultiNestOptimizer(path=path + 'test_files/non_linear/multinest/'
                                                              'optimizer/directory_setup/', model_mapper=model_map)

            assert os.path.exists(path + 'test_files/non_linear/multinest/optimizer/'
                                         'directory_setup/EllipticalSersic') == True
            assert multi.total_parameters == 7
            assert multi.generate_parameter_latex('x') == ['$x$']


    class TestMakeParamNames(object):

        def test__single_model_and_parameter_set__outputs_paramnames(self):

            light_profiles.EllipticalLightProfile._ids = count()

            if os.path.exists(path + 'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic'):
                shutil.rmtree(path + 'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic)

            non_linear.MultiNestOptimizer(path=path + 'test_files/non_linear/multinest/optimizer/make_param_names/', model_mapper=model_map)

            paramnames_test = open(path+'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic/model.paramnames')
            paramnames_str_0 = paramnames_test.readline()
            paramnames_str_1 = paramnames_test.readline()
            paramnames_str_2 = paramnames_test.readline()
            paramnames_str_3 = paramnames_test.readline()
            paramnames_str_4 = paramnames_test.readline()
            paramnames_str_5 = paramnames_test.readline()
            paramnames_str_6 = paramnames_test.readline()

            assert paramnames_str_0 == r'light_profile_0_centre_0                $x_{\mathrm{l1}}$'+'\n'
            assert paramnames_str_1 == r'light_profile_0_centre_1                $y_{\mathrm{l1}}$'+'\n'
            assert paramnames_str_2 == r'light_profile_0_axis_ratio              $q_{\mathrm{l1}}$'+'\n'
            assert paramnames_str_3 == r'light_profile_0_phi                     $\phi_{\mathrm{l1}}$'+'\n'
            assert paramnames_str_4 == r'light_profile_0_intensity               $I_{\mathrm{l1}}$'+'\n'
            assert paramnames_str_5 == r'light_profile_0_effective_radius        $R_{\mathrm{l1}}$'+'\n'
            assert paramnames_str_6 == r'light_profile_0_sersic_index            $n_{\mathrm{l1}}$'+'\n'

            shutil.rmtree(path + 'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic')

        def test__two_light_models_outputs_paramnames(self):

            light_profiles.EllipticalLightProfile._ids = count()

            if os.path.exists(path + 'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic+EllipticalExponential'):
                shutil.rmtree(path + 'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic+EllipticalExponential')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic,
                                                  light_profile_1=light_profiles.EllipticalExponential)

            non_linear.MultiNestOptimizer(path=path + 'test_files/non_linear/multinest/optimizer/make_param_names/', model_mapper=model_map)

            paramnames_test = open(path+'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic+EllipticalExponential/model.paramnames')
            paramnames_str = paramnames_test.readlines()

            assert paramnames_str[0] == r'light_profile_0_centre_0                $x_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[1] == r'light_profile_0_centre_1                $y_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[2] == r'light_profile_0_axis_ratio              $q_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[3] == r'light_profile_0_phi                     $\phi_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[4] == r'light_profile_0_intensity               $I_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[5] == r'light_profile_0_effective_radius        $R_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[6] == r'light_profile_0_sersic_index            $n_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[7] == r'light_profile_1_centre_0                $x_{\mathrm{l2}}$'+'\n'
            assert paramnames_str[8] == r'light_profile_1_centre_1                $y_{\mathrm{l2}}$'+'\n'
            assert paramnames_str[9] == r'light_profile_1_axis_ratio              $q_{\mathrm{l2}}$'+'\n'
            assert paramnames_str[10] == r'light_profile_1_phi                     $\phi_{\mathrm{l2}}$'+'\n'
            assert paramnames_str[11] == r'light_profile_1_intensity               $I_{\mathrm{l2}}$'+'\n'
            assert paramnames_str[12] == r'light_profile_1_effective_radius        $R_{\mathrm{l2}}$'+'\n'

            shutil.rmtree(path + 'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic+EllipticalExponential')

        def test__two_light_models__two_mass_models__outputs_paramnames(self):

            light_profiles.EllipticalLightProfile._ids = count()
            mass_profiles.EllipticalMassProfile._ids = count()

            if os.path.exists(path + 'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic+EllipticalExponential+'
                                     'SphericalIsothermal+SphericalNFW'):
                shutil.rmtree(path + 'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic+EllipticalExponential+'
                                     'SphericalIsothermal+SphericalNFW')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic,
                                                  light_profile_1=light_profiles.EllipticalExponential,
                                                  mass_profile_0=mass_profiles.SphericalIsothermal,
                                                  mass_profile_1=mass_profiles.SphericalNFW)

            non_linear.MultiNestOptimizer(path=path + 'test_files/non_linear/multinest/optimizer/make_param_names/', model_mapper=model_map)

            paramnames_test = open(path+'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic+EllipticalExponential+'
                                        'SphericalIsothermal+SphericalNFW/model.paramnames')
            paramnames_str = paramnames_test.readlines()

            assert paramnames_str[0] == r'light_profile_0_centre_0                $x_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[1] == r'light_profile_0_centre_1                $y_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[2] == r'light_profile_0_axis_ratio              $q_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[3] == r'light_profile_0_phi                     $\phi_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[4] == r'light_profile_0_intensity               $I_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[5] == r'light_profile_0_effective_radius        $R_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[6] == r'light_profile_0_sersic_index            $n_{\mathrm{l1}}$'+'\n'
            assert paramnames_str[7] == r'light_profile_1_centre_0                $x_{\mathrm{l2}}$'+'\n'
            assert paramnames_str[8] == r'light_profile_1_centre_1                $y_{\mathrm{l2}}$'+'\n'
            assert paramnames_str[9] == r'light_profile_1_axis_ratio              $q_{\mathrm{l2}}$'+'\n'
            assert paramnames_str[10] == r'light_profile_1_phi                     $\phi_{\mathrm{l2}}$'+'\n'
            assert paramnames_str[11] == r'light_profile_1_intensity               $I_{\mathrm{l2}}$'+'\n'
            assert paramnames_str[12] == r'light_profile_1_effective_radius        $R_{\mathrm{l2}}$'+'\n'
            assert paramnames_str[13] == r'mass_profile_0_centre_0                 $x_{\mathrm{1}}$'+'\n'
            assert paramnames_str[14] == r'mass_profile_0_centre_1                 $y_{\mathrm{1}}$'+'\n'
            assert paramnames_str[15] == r'mass_profile_0_einstein_radius          $\theta_{\mathrm{1}}$'+'\n'
            assert paramnames_str[16] == r'mass_profile_1_centre_0                 $x_{\mathrm{d2}}$'+'\n'
            assert paramnames_str[17] == r'mass_profile_1_centre_1                 $y_{\mathrm{d2}}$'+'\n'
            assert paramnames_str[18] == r'mass_profile_1_kappa_s                  $\kappa_{\mathrm{d2}}$'+'\n'
            assert paramnames_str[19] == r'mass_profile_1_scale_radius             $Rs_{\mathrm{d2}}$'+'\n'

            shutil.rmtree(path + 'test_files/non_linear/multinest/optimizer/make_param_names/EllipticalSersic+EllipticalExponential+'
                                 'SphericalIsothermal+SphericalNFW')


    class TestMakeModelInfo(object):

        def test__single_model__outputs_all_info(self):

            if os.path.exists(path + 'test_files/non_linear/multinest/optimizer/model_info/EllipticalSersic'):
                shutil.rmtree(path + 'test_files/non_linear/multinest/optimizer/model_info/EllipticalSersic')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic)

            non_linear.MultiNestOptimizer(path=path + 'test_files/non_linear/multinest/optimizer/model_info/', model_mapper=model_map)

            model_info_test = open(path + 'test_files/non_linear/multinest/optimizer/model_info/EllipticalSersic/model.info')
            model_info_str = model_info_test.readlines()

            assert model_info_str[0] == r'EllipticalSersic'+'\n'
            assert model_info_str[1] == r''+'\n'
            assert model_info_str[2] == r'centre_0: UniformPrior, lower_limit = 0.0, upper_limit = 1.0'+'\n'
            assert model_info_str[3] == r'centre_1: UniformPrior, lower_limit = 0.0, upper_limit = 0.5'+'\n'
            assert model_info_str[4] == r'axis_ratio: UniformPrior, lower_limit = 0.0, upper_limit = 0.5'+'\n'
            assert model_info_str[5] == r'phi: UniformPrior, lower_limit = 0.0, upper_limit = 0.5'+'\n'
            assert model_info_str[6] == r'intensity: GaussianPrior, mean = 0.0, sigma = 0.5'+'\n'
            assert model_info_str[7] == r'effective_radius: UniformPrior, lower_limit = 1.0, upper_limit = 1.0'+'\n'
            assert model_info_str[8] == r'sersic_index: UniformPrior, lower_limit = 1.0, upper_limit = 1.0'+'\n'

        def test__two_models_and_parameter_sets__outputs_paramnames(self):

            if os.path.exists(path + 'test_files/non_linear/multinest/optimizer/model_info/EllipticalSersic+EllipticalExponential+'
                                     'SphericalIsothermal+SphericalNFW'):
                shutil.rmtree(path + 'test_files/non_linear/multinest/optimizer/model_info/EllipticalSersic+EllipticalExponential+'
                                     'SphericalIsothermal+SphericalNFW')

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic,
                                                  light_profile_1=light_profiles.EllipticalExponential,
                                                  mass_profile_0=mass_profiles.SphericalIsothermal,
                                                  mass_profile_1=mass_profiles.SphericalNFW)

            non_linear.MultiNestOptimizer(path=path + 'test_files/non_linear/multinest/optimizer/model_info/', model_mapper=model_map)

            model_info_test = open(
                path + 'test_files/non_linear/multinest/optimizer/model_info/EllipticalSersic+EllipticalExponential+'
                       'SphericalIsothermal+SphericalNFW/model.info')
            model_info_str = model_info_test.readlines()

            assert model_info_str[0] == r'EllipticalSersic'+'\n'
            assert model_info_str[1] == r''+'\n'
            assert model_info_str[2] == r'centre_0: UniformPrior, lower_limit = 0.0, upper_limit = 1.0'+'\n'
            assert model_info_str[3] == r'centre_1: UniformPrior, lower_limit = 0.0, upper_limit = 0.5'+'\n'
            assert model_info_str[4] == r'axis_ratio: UniformPrior, lower_limit = 0.0, upper_limit = 0.5'+'\n'
            assert model_info_str[5] == r'phi: UniformPrior, lower_limit = 0.0, upper_limit = 0.5'+'\n'
            assert model_info_str[6] == r'intensity: GaussianPrior, mean = 0.0, sigma = 0.5'+'\n'
            assert model_info_str[7] == r'effective_radius: UniformPrior, lower_limit = 1.0, upper_limit = 1.0'+'\n'
            assert model_info_str[8] == r'sersic_index: UniformPrior, lower_limit = 1.0, upper_limit = 1.0'+'\n'
            assert model_info_str[9] == r''+'\n'
            assert model_info_str[10] == r'EllipticalExponential'+'\n'
            assert model_info_str[11] == r''+'\n'
            assert model_info_str[12] == r'centre_0: UniformPrior, lower_limit = 0.0, upper_limit = 1.0'+'\n'
            assert model_info_str[13] == r'centre_1: UniformPrior, lower_limit = 0.0, upper_limit = 0.5'+'\n'
            assert model_info_str[14] == r'axis_ratio: UniformPrior, lower_limit = 0.0, upper_limit = 0.5'+'\n'
            assert model_info_str[15] == r'phi: UniformPrior, lower_limit = 0.0, upper_limit = 0.5'+'\n'
            assert model_info_str[16] == r'intensity: GaussianPrior, mean = 0.0, sigma = 0.5'+'\n'
            assert model_info_str[17] == r'effective_radius: UniformPrior, lower_limit = 1.0, upper_limit = 1.0'+'\n'
            assert model_info_str[18] == r''+'\n'
            assert model_info_str[19] == r'SphericalIsothermal'+'\n'
            assert model_info_str[20] == r''+'\n'
            assert model_info_str[21] == r'centre_0: UniformPrior, lower_limit = 0.0, upper_limit = 1.0'+'\n'
            assert model_info_str[22] == r'centre_1: UniformPrior, lower_limit = 0.0, upper_limit = 0.5'+'\n'
            assert model_info_str[23] == r'einstein_radius: UniformPrior, lower_limit = 1.0, upper_limit = 1.0'+'\n'
            assert model_info_str[24] == r''+'\n'
            assert model_info_str[25] == r'SphericalNFW'+'\n'
            assert model_info_str[26] == r''+'\n'
            assert model_info_str[27] == r'centre_0: UniformPrior, lower_limit = 0.0, upper_limit = 1.0'+'\n'
            assert model_info_str[28] == r'centre_1: UniformPrior, lower_limit = 0.0, upper_limit = 0.5'+'\n'
            assert model_info_str[29] == r'kappa_s: UniformPrior, lower_limit = 1.0, upper_limit = 1.0'+'\n'
            assert model_info_str[30] == r'scale_radius: UniformPrior, lower_limit = 1.0, upper_limit = 1.0'+'\n'


    class TestCheckModelInfo(object):

        def test__single_model__prior_changed_from_input_model__raises_error(self):
            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic)

            with pytest.raises(non_linear.MultiNestException):
                non_linear.MultiNestOptimizer(path=path + 'test_files/non_linear/multinest/optimizer/check_model_info/', model_mapper=model_map)


class TestMultiNestResults(object):


    class TestLoadModels(object):

        def test__one_profile__read_most_probable_vector__via_summary(self):

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestResults(path=path + 'test_files/non_linear/multinest/results/summaries/',
                                                  model_mapper=model_map)

            assert results._most_probable == [1.0, 2.0, 3.0, 4.0]

        def test__multiple_profile__read_most_probable_vector__via_summary(self):

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                light_profile=light_profiles.EllipticalSersic,
                                                mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestResults(path=path + 'test_files/non_linear/multinest/results/summaries/',
                                                  model_mapper=model_map)

            assert results._most_probable == [1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0, 11.0]

        def test__one_profile__read_most_likely_vector__via_summary(self):

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestResults(path=path + 'test_files/non_linear/multinest/results/summaries/',
                                                  model_mapper=model_map)

            assert results._most_likely == [5.0, 6.0, 7.0, 8.0]

        def test__multiple_profile__read_most_likely_vector__via_summary(self):

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config,
                                                light_profile=light_profiles.EllipticalSersic,
                                                mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestResults(path=path + 'test_files/non_linear/multinest/results/summaries/',
                                                  model_mapper=model_map)

            assert results._most_likely == [12.0, 13.0, 14.0, 15.0, 16.0, -17.0, -18.0, -19.0, -20.0, 21.0, 22.0]


    class TestSetupModelInstances(object):

        def test__one_profile__setup_most_probable_model(self):

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestResults(path=path + 'test_files/non_linear/multinest/results/summaries/',
                                                  model_mapper=model_map)

            assert results.most_probable.mass_profile.centre == (1.0, 2.0)
            assert results.most_probable.mass_profile.kappa_s == 3.0
            assert results.most_probable.mass_profile.scale_radius == 4.0

        def test__multiple_profiles__setup_most_probable_model(self):

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(
                config=config, light_profile=light_profiles.EllipticalSersic, mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestResults(path=path + 'test_files/non_linear/multinest/results/summaries/',
                                                  model_mapper=model_map)

            assert results.most_probable.light_profile.centre == (1.0, 2.0)
            assert results.most_probable.light_profile.axis_ratio == 3.0
            assert results.most_probable.light_profile.phi == 4.0
            assert results.most_probable.light_profile.intensity == -5.0
            assert results.most_probable.light_profile.effective_radius == -6.0
            assert results.most_probable.light_profile.sersic_index == -7.0

            assert results.most_probable.mass_profile.centre == (-8.0, 9.0)
            assert results.most_probable.mass_profile.kappa_s == 10.0
            assert results.most_probable.mass_profile.scale_radius == 11.0

        def test__one_profile__setup_most_likely_model(self):

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestResults(path=path + 'test_files/non_linear/multinest/results/summaries/',
                                                  model_mapper=model_map)

            assert results.most_likely.mass_profile.centre == (5.0, 6.0)
            assert results.most_likely.mass_profile.kappa_s == 7.0
            assert results.most_likely.mass_profile.scale_radius == 8.0

        def test__multiple_profile__read_most_likely_vector__via_summary(self):

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(
                config=config, light_profile=light_profiles.EllipticalSersic, mass_profile=mass_profiles.SphericalNFW)

            results = non_linear.MultiNestResults(path=path + 'test_files/non_linear/multinest/results/summaries/',
                                                  model_mapper=model_map)

            assert results.most_likely.light_profile.centre == (12.0, 13.0)
            assert results.most_likely.light_profile.axis_ratio == 14.0
            assert results.most_likely.light_profile.phi == 15.0
            assert results.most_likely.light_profile.intensity == 16.0
            assert results.most_likely.light_profile.effective_radius == -17.0
            assert results.most_likely.light_profile.sersic_index == -18.0

            assert results.most_likely.mass_profile.centre == (-19.0, -20.0)
            assert results.most_likely.mass_profile.kappa_s == 21.0
            assert results.most_likely.mass_profile.scale_radius == 22.0


    class TestSetupFromMultiNest(object):

        def test__setup_from_multinest__identical_to_results_class(self):

            config = model_mapper.Config(config_folder_path=path + 'test_files/config')
            model_map = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

            multi = non_linear.MultiNestOptimizer(path=path + 'test_files/non_linear/multinest/results/summaries/',
                                                  model_mapper=model_map)

            results_1 = multi.setup_results()

            results_2 = non_linear.MultiNestResults(path=path + 'test_files/non_linear/multinest/results/summaries/',
                                                  model_mapper=model_map)



            assert results_1._most_probable == results_2._most_probable
            assert results_1._most_likely == results_2._most_likely