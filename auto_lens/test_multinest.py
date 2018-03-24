import os
import shutil
from auto_lens import multinest
from auto_lens import model_mapper
from auto_lens.profiles import geometry_profiles, light_profiles, mass_profiles

path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))

class TestDirectorySetup(object):

    def test__one_light_profile__correct_directory(self):

        if os.path.exists(path + 'test_files/multinest/directory_setup/EllipticalSersic'):
            shutil.rmtree(path + 'test_files/multinest/directory_setup/EllipticalSersic')

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        collection = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic)

        multinest.MultiNest(path=path + 'test_files/multinest/directory_setup/', model=collection)

        assert os.path.exists(path + 'test_files/multinest/directory_setup/EllipticalSersic') == True

    def test__one_mass_profile__correct_directory(self):
        
        if os.path.exists(path + 'test_files/multinest/directory_setup/SphericalNFW'):
            shutil.rmtree(path + 'test_files/multinest/directory_setup/SphericalNFW')

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        collection = model_mapper.ModelMapper(config=config, mass_profile=mass_profiles.SphericalNFW)

        multinest.MultiNest(path=path + 'test_files/multinest/directory_setup/', model=collection)

        assert os.path.exists(path + 'test_files/multinest/directory_setup/SphericalNFW') == True

    def test__multiple_light_and_mass_profiles__correct_directory(self):
        
        if os.path.exists(path +
                'test_files/multinest/directory_setup/EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW'):
            shutil.rmtree(
                path + 'test_files/multinest/directory_setup/EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW')

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        collection = model_mapper.ModelMapper(config=config, light_profile=light_profiles.EllipticalSersic,
                                              light_profile_2=light_profiles.EllipticalSersic,
                                              light_profile_3=light_profiles.EllipticalSersic,
                                              mass_profile=mass_profiles.SphericalNFW,
                                              mass_profile_2=mass_profiles.SphericalNFW)

        multinest.MultiNest(path=path + 'test_files/multinest/directory_setup/', model=collection)

        assert os.path.exists(path +
                'test_files/multinest/directory_setup/EllipticalSersic+EllipticalSersic+EllipticalSersic+SphericalNFW+SphericalNFW') == True


class TestGenerateLatex(object):

    def test__one_parameter__no_subscript(self):

        assert multinest.generate_parameter_latex('x') == ['$x$']

    def test__three_parameters__no_subscript(self):

        assert multinest.generate_parameter_latex(['x', 'y', 'z']) == ['$x$', '$y$', '$z$']

    def test__one_parameter__subscript__no_number(self):

        assert multinest.generate_parameter_latex(['x'], subscript='d') == [r'$x_{\mathrm{d}}$']

    def test__three_parameters__subscript__no_number(self):

        assert multinest.generate_parameter_latex(['x', 'y', 'z'], subscript='d') == [r'$x_{\mathrm{d}}$',
                                                                                    r'$y_{\mathrm{d}}$',
                                                                                    r'$z_{\mathrm{d}}$']


class TestMakeParamNames(object):

    def test__single_model_and_parameter_set__outputs_paramnames(self):

        if os.path.exists(path + 'test_files/multinest/make_param_names/EllipticalSersic'):
            shutil.rmtree(path + 'test_files/multinest/make_param_names/EllipticalSersic')

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        collection = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic)

        multinest.MultiNest(path=path + 'test_files/multinest/make_param_names/', model=collection)

        paramnames_test = open(path+'test_files/multinest/make_param_names/EllipticalSersic/model.paramnames')
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

        shutil.rmtree(path + 'test_files/multinest/make_param_names/EllipticalSersic')

    def test__two_light_models_outputs_paramnames(self):

        if os.path.exists(path + 'test_files/multinest/make_param_names/EllipticalSersic+EllipticalExponential'):
            shutil.rmtree(path + 'test_files/multinest/make_param_names/EllipticalSersic+EllipticalExponential')

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        collection = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic,
                                              light_profile_1=light_profiles.EllipticalExponential)

        multinest.MultiNest(path=path + 'test_files/multinest/make_param_names/', model=collection)

        paramnames_test = open(path+'test_files/multinest/make_param_names/EllipticalSersic+EllipticalExponential/model.paramnames')
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

        shutil.rmtree(path + 'test_files/multinest/make_param_names/EllipticalSersic+EllipticalExponential')

    def test__two_light_models__two_mass_models__outputs_paramnames(self):

        if os.path.exists(path + 'test_files/multinest/make_param_names/EllipticalSersic+EllipticalExponential+'
                                 'SphericalIsothermal+SphericalNFW'):
            shutil.rmtree(path + 'test_files/multinest/make_param_names/EllipticalSersic+EllipticalExponential+'
                                 'SphericalIsothermal+SphericalNFW')

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        collection = model_mapper.ModelMapper(config=config, light_profile_0=light_profiles.EllipticalSersic,
                                              light_profile_1=light_profiles.EllipticalExponential,
                                              mass_profile_0=mass_profiles.SphericalIsothermal,
                                              mass_profile_1=mass_profiles.SphericalNFW)

        multinest.MultiNest(path=path + 'test_files/multinest/make_param_names/', model=collection)

        paramnames_test = open(path+'test_files/multinest/make_param_names/EllipticalSersic+EllipticalExponential+'
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

        shutil.rmtree(path + 'test_files/multinest/make_param_names/EllipticalSersic+EllipticalExponential+'
                             'SphericalIsothermal+SphericalNFW')

class TestLoadModels(object):

    def test__one_profile__read_most_probable_vector__via_summary(self):

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        modeling = model_mapper.ModelMapper(config=config, geometry_profile=geometry_profiles.EllipticalProfile)

        multi = multinest.MultiNest(path=path+'test_files/multinest/summaries/', model=modeling)
        most_probable_vector = multi.read_most_probable()

        assert most_probable_vector == [1.0, 2.0, 3.0, 4.0]

    def test__multiple_profile__read_most_probable_vector__via_summary(self):

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        modeling = model_mapper.ModelMapper(config=config, geometry_profile=geometry_profiles.EllipticalProfile,
                                            light_profile=light_profiles.EllipticalSersic,
                                            mass_profile=mass_profiles.SphericalNFW)

        multi = multinest.MultiNest(path=path+'test_files/multinest/summaries/', model=modeling)
        most_probable_vector = multi.read_most_probable()

        assert most_probable_vector == [1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0, 11.0, 12.0]

    def test__one_profile__read_most_likely_vector__via_summary(self):

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        modeling = model_mapper.ModelMapper(config=config, geometry_profile=geometry_profiles.EllipticalProfile)

        multi = multinest.MultiNest(path=path+'test_files/multinest/summaries/', model=modeling)
        most_likely_vector = multi.read_most_likely()

        assert most_likely_vector == [5.0, 6.0, 7.0, 8.0]

    def test__multiple_profile__read_most_likely_vector__via_summary(self):

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        modeling = model_mapper.ModelMapper(config=config, geometry_profile=geometry_profiles.EllipticalProfile,
                                            light_profile=light_profiles.EllipticalSersic,
                                            mass_profile=mass_profiles.SphericalNFW)

        multi = multinest.MultiNest(path=path+'test_files/multinest/summaries/', model=modeling)
        most_likely_vector = multi.read_most_likely()
        assert most_likely_vector == [13.0, 14.0, 15.0, 16.0, -17.0, -18.0, -19.0, -20.0, 21.0, 22.0, 23.0, 24.0]


class TestSetupModelInstances(object):

    def test__one_profile__setup_most_probable_model(self):

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        modeling = model_mapper.ModelMapper(config=config, geometry_profile=geometry_profiles.EllipticalProfile)

        multi = multinest.MultiNest(path=path+'test_files/multinest/summaries/', model=modeling)
        model_instance = multi.setup_most_probable_model_instance()

        assert model_instance.geometry_profile.centre == (1.0, 2.0)
        assert model_instance.geometry_profile.axis_ratio == 3.0
        assert model_instance.geometry_profile.phi == 4.0

    def test__multiple_profiles__setup_most_probable_model(self):

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        modeling = model_mapper.ModelMapper(
            config=config,
            profile_1=geometry_profiles.EllipticalProfile, profile_2=geometry_profiles.EllipticalProfile,
            profile_3=geometry_profiles.EllipticalProfile)

        multi = multinest.MultiNest(path=path+'test_files/multinest/summaries/', model=modeling)
        model_instance = multi.setup_most_probable_model_instance()

        assert model_instance.profile_1.centre == (1.0, 2.0)
        assert model_instance.profile_1.axis_ratio == 3.0
        assert model_instance.profile_1.phi == 4.0

        assert model_instance.profile_2.centre == (-5.0, -6.0)
        assert model_instance.profile_2.axis_ratio == -7.0
        assert model_instance.profile_2.phi == -8.0

        assert model_instance.profile_3.centre == (9.0, 10.0)
        assert model_instance.profile_3.axis_ratio == 11.0
        assert model_instance.profile_3.phi == 12.0

    # TODO : Another example of setitng different parameters to one another messing up the vector.

    def test__multiple_profiles__set_one_parameter_to_another__read_most_probable_vector__via_summary(self):

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        modeling = model_mapper.ModelMapper(config=config, geometry_profile=geometry_profiles.EllipticalProfile,
                                            light_profile=light_profiles.EllipticalSersic,
                                            mass_profile=mass_profiles.SphericalNFW)

        modeling.mass_profile.centre = modeling.light_profile.centre

        multi = multinest.MultiNest(path=path+'test_files/multinest/summaries/', model=modeling)
        model_instance = multi.setup_most_probable_model_instance()

        assert model_instance.geometry_profile.centre == (1.0, 2.0)
        assert model_instance.geometry_profile.axis_ratio == 3.0
        assert model_instance.geometry_profile.phi == 4.0

        assert model_instance.light_profile.centre == (-5.0, -6.0)
        assert model_instance.light_profile.axis_ratio == -7.0
    #    assert model_instance.light_profile.phi == -8.0
        assert model_instance.light_profile.intensity == 9.0
        assert model_instance.light_profile.effective_radius == 10.0
        assert model_instance.light_profile.sersic_index == 11.0

        assert model_instance.mass_profile.kappa_s == 12.0

    def test__one_profile__read_most_likely_vector__via_summary(self):

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        modeling = model_mapper.ModelMapper(config=config, geometry_profile=geometry_profiles.EllipticalProfile)

        multi = multinest.MultiNest(path=path+'test_files/multinest/summaries/', model=modeling)
        most_likely_vector = multi.read_most_likely()

        assert most_likely_vector == [5.0, 6.0, 7.0, 8.0]

    def test__multiple_profile__read_most_likely_vector__via_summary(self):

        config = model_mapper.Config(config_folder_path=path + 'test_files/config')
        modeling = model_mapper.ModelMapper(config=config, geometry_profile=geometry_profiles.EllipticalProfile,
                                            light_profile=light_profiles.EllipticalSersic,
                                            mass_profile=mass_profiles.SphericalNFW)

        multi = multinest.MultiNest(path=path+'test_files/multinest/summaries/', model=modeling)
        most_likely_vector = multi.read_most_likely()
        assert most_likely_vector == [13.0, 14.0, 15.0, 16.0, -17.0, -18.0, -19.0, -20.0, 21.0, 22.0, 23.0, 24.0]