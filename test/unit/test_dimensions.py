import pytest

from autolens import exc, dimensions as dim

from test.unit.mock.mock_cosmology import MockCosmology


class TestLength(object):

    def test__conversions_from_arcsec_to_kpc_and_back__errors_raised_if_no_kpc_per_arcsec(self):
        unit_arcsec = dim.Length(value=2.0)

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit_length == 'arcsec'

        unit_arcsec = unit_arcsec.convert(unit_length='arcsec')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == 'arcsec'

        unit_kpc = unit_arcsec.convert(unit_length='kpc', kpc_per_arcsec=2.0)

        assert unit_kpc == 4.0
        assert unit_kpc.unit == 'kpc'

        unit_kpc = unit_kpc.convert(unit_length='kpc')

        assert unit_kpc == 4.0
        assert unit_kpc.unit == 'kpc'

        unit_arcsec = unit_kpc.convert(unit_length='arcsec', kpc_per_arcsec=2.0)

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == 'arcsec'

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit_length='kpc')
            unit_kpc.convert(unit_length='arcsec')
            unit_arcsec.convert(unit_length='lol')


class TestLuminosity(object):

    def test__conversions_from_eps_and_counts_and_back__errors_raised_if_no_exposure_time(self):

        unit_eps = dim.Luminosity(value=2.0)

        assert unit_eps == 2.0
        assert unit_eps.unit_luminosity == 'eps'

        unit_eps = unit_eps.convert(unit_luminosity='eps')

        assert unit_eps == 2.0
        assert unit_eps.unit == 'eps'

        unit_counts = unit_eps.convert(unit_luminosity='counts', exposure_time=2.0)

        assert unit_counts == 4.0
        assert unit_counts.unit == 'counts'

        unit_counts = unit_counts.convert(unit_luminosity='counts')

        assert unit_counts == 4.0
        assert unit_counts.unit == 'counts'

        unit_eps = unit_counts.convert(unit_luminosity='eps', exposure_time=2.0)

        assert unit_eps == 2.0
        assert unit_eps.unit == 'eps'

        with pytest.raises(exc.UnitsException):
            unit_eps.convert(unit_luminosity='counts')
            unit_counts.convert(unit_luminosity='eps')
            unit_eps.convert(unit_luminosity='lol')


class TestMass(object):

    def test__conversions_from_angular_and_sol_mass_and_back__errors_raised_if_no_exposure_time(self):

        mass_angular = dim.Mass(value=2.0)

        assert mass_angular == 2.0
        assert mass_angular.unit_mass == 'angular'

        # angular -> angular, stays 2.0

        mass_angular = mass_angular.convert(unit_mass='angular')

        assert mass_angular == 2.0
        assert mass_angular.unit == 'angular'

        # angular -> solMass, converts to 2.0 * 2.0 = 4.0

        mas_sol_mass = mass_angular.convert(unit_mass='solMass',critical_surface_density=2.0)

        assert mas_sol_mass == 4.0
        assert mas_sol_mass.unit == 'solMass'

        # solMass -> solMass, stays 4.0

        mas_sol_mass = mas_sol_mass.convert(unit_mass='solMass')

        assert mas_sol_mass == 4.0
        assert mas_sol_mass.unit == 'solMass'

        # solMass -> angular, stays 4.0

        mass_angular = mas_sol_mass.convert(unit_mass='angular', critical_surface_density=2.0)

        assert mass_angular == 2.0
        assert mass_angular.unit == 'angular'

        with pytest.raises(exc.UnitsException):
            mass_angular.convert(unit_mass='solMass')
            mas_sol_mass.convert(unit_mass='angular')
            mass_angular.convert(unit_mass='lol')


class TestMassOverLuminosity(object):

    def test__conversions_from_angular_and_sol_mass_and_back__errors_raised_if_critical_mass_density(self):

        unit_angular = dim.MassOverLuminosity(value=2.0)

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular / eps'

        unit_angular = unit_angular.convert(unit_mass='angular', unit_luminosity='eps')

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular / eps'

        unit_sol_mass = unit_angular.convert(unit_mass='solMass', critical_surface_density=2.0,
                                             unit_luminosity='eps')

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == 'solMass / eps'

        unit_sol_mass = unit_sol_mass.convert(unit_mass='solMass', unit_luminosity='eps')

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == 'solMass / eps'

        unit_angular = unit_sol_mass.convert(unit_mass='angular', critical_surface_density=2.0,
                                             unit_luminosity='eps')

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular / eps'

        with pytest.raises(exc.UnitsException):
            unit_angular.convert(unit_mass='solMass', unit_luminosity='eps')
            unit_sol_mass.convert(unit_mass='angular', unit_luminosity='eps')
            unit_angular.convert(unit_mass='lol', unit_luminosity='eps')
            
    def test__conversions_from_eps_and_counts_and_back__errors_raised_if_no_exposure_time(self):

        unit_eps = dim.MassOverLuminosity(value=2.0)

        assert unit_eps == 2.0
        assert unit_eps.unit == 'angular / eps'

        unit_eps = unit_eps.convert(unit_mass='angular', unit_luminosity='eps')

        assert unit_eps == 2.0
        assert unit_eps.unit == 'angular / eps'

        unit_counts = unit_eps.convert(unit_mass='angular', exposure_time=2.0, unit_luminosity='counts')

        assert unit_counts == 1.0
        assert unit_counts.unit == 'angular / counts'

        unit_counts = unit_counts.convert(unit_mass='angular', unit_luminosity='counts')

        assert unit_counts == 1.0
        assert unit_counts.unit == 'angular / counts'

        unit_eps = unit_counts.convert(unit_mass='angular', exposure_time=2.0, unit_luminosity='eps')

        assert unit_eps == 2.0
        assert unit_eps.unit == 'angular / eps'

        with pytest.raises(exc.UnitsException):
            unit_eps.convert(unit_mass='angular', unit_luminosity='eps')
            unit_counts.convert(unit_mass='angular', unit_luminosity='eps')
            unit_eps.convert(unit_mass='lol', unit_luminosity='eps')


class TestMassOverLength2(object):

    def test__conversions_from_angular_and_sol_mass_and_back__errors_raised_if_critical_mass_density(self):

        unit_angular = dim.MassOverLength2(value=2.0)

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular / arcsec^2'

        unit_angular = unit_angular.convert(unit_mass='angular', unit_length='arcsec')

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular / arcsec^2'

        unit_sol_mass = unit_angular.convert(unit_mass='solMass', critical_surface_density=2.0,
                                             unit_length='arcsec')

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == 'solMass / arcsec^2'

        unit_sol_mass = unit_sol_mass.convert(unit_mass='solMass', unit_length='arcsec')

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == 'solMass / arcsec^2'

        unit_angular = unit_sol_mass.convert(unit_mass='angular', critical_surface_density=2.0,
                                             unit_length='arcsec')

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular / arcsec^2'

        with pytest.raises(exc.UnitsException):
            unit_angular.convert(unit_mass='solMass', unit_length='eps')
            unit_sol_mass.convert(unit_mass='angular', unit_length='eps')
            unit_angular.convert(unit_mass='lol', unit_length='eps')

    def test__conversions_from_arcsec_to_kpc_and_back__errors_raised_if_no_kpc_per_arcsec(self):

        unit_arcsec = dim.MassOverLength2(value=2.0, unit_mass='solMass')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == 'solMass / arcsec^2'

        unit_arcsec = unit_arcsec.convert(unit_length='arcsec', unit_mass='solMass')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == 'solMass / arcsec^2'

        unit_kpc = unit_arcsec.convert(unit_length='kpc', kpc_per_arcsec=2.0, unit_mass='solMass')

        assert unit_kpc == 2.0 / 2.0**2.0
        assert unit_kpc.unit == 'solMass / kpc^2'

        unit_kpc = unit_kpc.convert(unit_length='kpc', unit_mass='solMass')

        assert unit_kpc == 2.0 / 2.0**2.0
        assert unit_kpc.unit == 'solMass / kpc^2'

        unit_arcsec = unit_kpc.convert(unit_length='arcsec', kpc_per_arcsec=2.0, unit_mass='solMass')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == 'solMass / arcsec^2'

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit_length='kpc', unit_mass='solMass')
            unit_kpc.convert(unit_length='arcsec', unit_mass='solMass')
            unit_arcsec.convert(unit_length='lol', unit_mass='solMass')


class TestMassOverLength3(object):

    def test__conversions_from_angular_and_sol_mass_and_back__errors_raised_if_critical_mass_density(self):

        unit_angular = dim.MassOverLength3(value=2.0)

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular / arcsec^3'

        unit_angular = unit_angular.convert(unit_mass='angular', unit_length='arcsec')

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular / arcsec^3'

        unit_sol_mass = unit_angular.convert(unit_mass='solMass', critical_surface_density=2.0,
                                             unit_length='arcsec')

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == 'solMass / arcsec^3'

        unit_sol_mass = unit_sol_mass.convert(unit_mass='solMass', unit_length='arcsec')

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == 'solMass / arcsec^3'

        unit_angular = unit_sol_mass.convert(unit_mass='angular', critical_surface_density=2.0,
                                             unit_length='arcsec')

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular / arcsec^3'

        with pytest.raises(exc.UnitsException):
            unit_angular.convert(unit_mass='solMass', unit_length='eps')
            unit_sol_mass.convert(unit_mass='angular', unit_length='eps')
            unit_angular.convert(unit_mass='lol', unit_length='eps')

    def test__conversions_from_arcsec_to_kpc_and_back__errors_raised_if_no_kpc_per_arcsec(self):

        unit_arcsec = dim.MassOverLength3(value=2.0, unit_mass='solMass')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == 'solMass / arcsec^3'

        unit_arcsec = unit_arcsec.convert(unit_length='arcsec', unit_mass='solMass')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == 'solMass / arcsec^3'

        unit_kpc = unit_arcsec.convert(unit_length='kpc', kpc_per_arcsec=2.0, unit_mass='solMass')

        assert unit_kpc == 2.0 / 2.0**3.0
        assert unit_kpc.unit == 'solMass / kpc^3'

        unit_kpc = unit_kpc.convert(unit_length='kpc', unit_mass='solMass')

        assert unit_kpc == 2.0 / 2.0**3.0
        assert unit_kpc.unit == 'solMass / kpc^3'

        unit_arcsec = unit_kpc.convert(unit_length='arcsec', kpc_per_arcsec=2.0, unit_mass='solMass')

        assert unit_arcsec == 2.0
        assert unit_arcsec.unit == 'solMass / arcsec^3'

        with pytest.raises(exc.UnitsException):
            unit_arcsec.convert(unit_length='kpc', unit_mass='solMass')
            unit_kpc.convert(unit_length='arcsec', unit_mass='solMass')
            unit_arcsec.convert(unit_length='lol', unit_mass='solMass')


class MockDimensionsProfile(dim.DimensionsProfile):

    def __init__(self,
                 position: dim.Position = None,
                 param_float: float = None,
                 length: dim.Length = None,
                 luminosity : dim.Luminosity = None,
                 mass : dim.Mass = None,
                 mass_over_luminosity: dim.MassOverLuminosity = None):

        super(MockDimensionsProfile, self).__init__()

        self.position = position
        self.param_float = param_float
        self.luminosity = luminosity
        self.length = length
        self.mass = mass
        self.mass_over_luminosity = mass_over_luminosity

    @dim.convert_units_to_input_units
    def unit_length_calc(self, length_input : dim.Length, redshift_profile=None, cosmology=MockCosmology(),
                         unit_length='arcsec', **kwargs):

        return dim.Length(self.length + length_input, self.unit_length)

    @dim.convert_units_to_input_units
    def unit_luminosity_calc(self,
                             luminosity_input : dim.Luminosity = None, redshift_profile=None, cosmology=MockCosmology(),
                             unit_luminosity='eps',
                             exposure_time : float = None, **kwargs):

        return dim.Luminosity(self.luminosity + luminosity_input, self.unit_luminosity)

    @dim.convert_units_to_input_units
    def unit_mass_calc(self,
                       mass_input : dim.Mass = None, redshift_profile=None, redshift_source=1.0, cosmology=MockCosmology(),
                       unit_mass='angular', **kwargs):

        return dim.Mass(self.mass + mass_input, self.unit_mass)


class TestDimensionsProfile(object):

    class TestUnitProperties(object):

        def test__unit_length__extracted_from_profile(self):

            profile = MockDimensionsProfile(length=dim.Length(value=3.0, unit_length='arcsec'))

            assert profile.unit_length == 'arcsec'

            profile = MockDimensionsProfile(length=dim.Length(value=3.0, unit_length='kpc'))

            assert profile.unit_length == 'kpc'

            profile = MockDimensionsProfile(length=1.0)

            assert profile.unit_length == None

        def test__unit_luminosity__extracted_from_profile(self):

            profile = MockDimensionsProfile(luminosity=dim.Luminosity(value=3.0, unit_luminosity='eps'),
                                            mass_over_luminosity=dim.MassOverLuminosity(value=3.0, unit_luminosity='eps'))

            assert profile.unit_luminosity == 'eps'

            profile = MockDimensionsProfile(luminosity=1.0,
                                            mass_over_luminosity=dim.MassOverLuminosity(value=3.0, unit_luminosity='eps'))

            assert profile.unit_luminosity == 'eps'

            profile = MockDimensionsProfile(luminosity=dim.Luminosity(value=3.0, unit_luminosity='eps'),
                                            mass_over_luminosity=1.0)

            assert profile.unit_luminosity == 'eps'

            profile = MockDimensionsProfile(luminosity=dim.Luminosity(value=3.0, unit_luminosity='counts'),
                                            mass_over_luminosity=dim.MassOverLuminosity(value=3.0, unit_luminosity='counts'))

            assert profile.unit_luminosity == 'counts'

            profile = MockDimensionsProfile(luminosity=1.0,
                                            mass_over_luminosity=dim.MassOverLuminosity(value=3.0, unit_luminosity='counts'))

            assert profile.unit_luminosity == 'counts'


            profile = MockDimensionsProfile(luminosity=dim.Luminosity(value=3.0, unit_luminosity='counts'),
                                            mass_over_luminosity=1.0)

            assert profile.unit_luminosity == 'counts'

            profile = MockDimensionsProfile(luminosity=1.0, mass_over_luminosity=1.0)

            assert profile.unit_luminosity == None
            
        def test__unit_mass__extracted_from_profile(self):

            profile = MockDimensionsProfile(mass=dim.Mass(value=3.0, unit_mass='angular'),
                                            mass_over_luminosity=dim.MassOverLuminosity(value=3.0, unit_mass='angular'))

            assert profile.unit_mass == 'angular'

            profile = MockDimensionsProfile(mass=1.0,
                                            mass_over_luminosity=dim.MassOverLuminosity(value=3.0, unit_mass='angular'))

            assert profile.unit_mass == 'angular'

            profile = MockDimensionsProfile(mass=dim.Mass(value=3.0, unit_mass='angular'),
                                            mass_over_luminosity=1.0)

            assert profile.unit_mass == 'angular'

            profile = MockDimensionsProfile(mass=dim.Mass(value=3.0, unit_mass='solMass'),
                                            mass_over_luminosity=dim.MassOverLuminosity(value=3.0, unit_mass='solMass'))

            assert profile.unit_mass == 'solMass'

            profile = MockDimensionsProfile(mass=1.0,
                                            mass_over_luminosity=dim.MassOverLuminosity(value=3.0, unit_mass='solMass'))

            assert profile.unit_mass == 'solMass'


            profile = MockDimensionsProfile(mass=dim.Mass(value=3.0, unit_mass='solMass'),
                                            mass_over_luminosity=1.0)

            assert profile.unit_mass == 'solMass'

            profile = MockDimensionsProfile(mass=1.0, mass_over_luminosity=1.0)

            assert profile.unit_mass == None


    class TestUnitConversions(object):

        def test__arcsec_to_kpc_conversions_of_length__float_and_tuple_length__conversion_converts_values(self):

            profile_arcsec = MockDimensionsProfile(
                position=(dim.Length(1.0, 'arcsec'), dim.Length(2.0, 'arcsec')),
                param_float=2.0,
                length=dim.Length(value=3.0, unit_length='arcsec'),
                luminosity=dim.Luminosity(value=4.0, unit_luminosity='eps'),
                mass=dim.Mass(value=5.0, unit_mass='angular'),
                mass_over_luminosity=dim.MassOverLuminosity(value=6.0, unit_luminosity='eps', unit_mass='angular'))

            assert profile_arcsec.position == (1.0, 2.0)
            assert profile_arcsec.position[0].unit_length == 'arcsec'
            assert profile_arcsec.position[1].unit_length == 'arcsec'
            assert profile_arcsec.param_float == 2.0
            assert profile_arcsec.length == 3.0
            assert profile_arcsec.length.unit_length == 'arcsec'
            assert profile_arcsec.luminosity == 4.0
            assert profile_arcsec.luminosity.unit_luminosity == 'eps'
            assert profile_arcsec.mass == 5.0
            assert profile_arcsec.mass.unit_mass == 'angular'
            assert profile_arcsec.mass_over_luminosity == 6.0
            assert profile_arcsec.mass_over_luminosity.unit == 'angular / eps'

            profile_arcsec = profile_arcsec.new_profile_with_units_converted(unit_length='arcsec')

            assert profile_arcsec.position == (1.0, 2.0)
            assert profile_arcsec.position[0].unit == 'arcsec'
            assert profile_arcsec.position[1].unit == 'arcsec'
            assert profile_arcsec.param_float == 2.0
            assert profile_arcsec.length == 3.0
            assert profile_arcsec.length.unit == 'arcsec'
            assert profile_arcsec.luminosity == 4.0
            assert profile_arcsec.luminosity.unit == 'eps'
            assert profile_arcsec.mass == 5.0
            assert profile_arcsec.mass.unit_mass == 'angular'
            assert profile_arcsec.mass_over_luminosity == 6.0
            assert profile_arcsec.mass_over_luminosity.unit == 'angular / eps'

            profile_kpc = profile_arcsec.new_profile_with_units_converted(unit_length='kpc', kpc_per_arcsec=2.0)

            assert profile_kpc.position == (2.0, 4.0)
            assert profile_kpc.position[0].unit == 'kpc'
            assert profile_kpc.position[1].unit == 'kpc'
            assert profile_kpc.param_float == 2.0
            assert profile_kpc.length == 6.0
            assert profile_kpc.length.unit == 'kpc'
            assert profile_kpc.luminosity == 4.0
            assert profile_kpc.luminosity.unit == 'eps'
            assert profile_arcsec.mass == 5.0
            assert profile_arcsec.mass.unit_mass == 'angular'
            assert profile_kpc.mass_over_luminosity == 6.0
            assert profile_kpc.mass_over_luminosity.unit == 'angular / eps'

            profile_kpc = profile_kpc.new_profile_with_units_converted(unit_length='kpc')

            assert profile_kpc.position == (2.0, 4.0)
            assert profile_kpc.position[0].unit == 'kpc'
            assert profile_kpc.position[1].unit == 'kpc'
            assert profile_kpc.param_float == 2.0
            assert profile_kpc.length == 6.0
            assert profile_kpc.length.unit == 'kpc'
            assert profile_kpc.luminosity == 4.0
            assert profile_kpc.luminosity.unit == 'eps'
            assert profile_arcsec.mass == 5.0
            assert profile_arcsec.mass.unit_mass == 'angular'
            assert profile_kpc.mass_over_luminosity == 6.0
            assert profile_kpc.mass_over_luminosity.unit == 'angular / eps'

            profile_arcsec = profile_kpc.new_profile_with_units_converted(unit_length='arcsec', kpc_per_arcsec=2.0)

            assert profile_arcsec.position == (1.0, 2.0)
            assert profile_arcsec.position[0].unit == 'arcsec'
            assert profile_arcsec.position[1].unit == 'arcsec'
            assert profile_arcsec.param_float == 2.0
            assert profile_arcsec.length == 3.0
            assert profile_arcsec.length.unit == 'arcsec'
            assert profile_arcsec.luminosity == 4.0
            assert profile_arcsec.luminosity.unit == 'eps'
            assert profile_arcsec.mass == 5.0
            assert profile_arcsec.mass.unit_mass == 'angular'
            assert profile_arcsec.mass_over_luminosity == 6.0
            assert profile_arcsec.mass_over_luminosity.unit == 'angular / eps'

        def test__conversion_requires_kpc_per_arcsec_but_does_not_supply_it_raises_error(self):

            profile_arcsec = MockDimensionsProfile(position=(dim.Length(1.0, 'arcsec'), dim.Length(2.0, 'arcsec')),)

            with pytest.raises(exc.UnitsException):
                profile_arcsec.new_profile_with_units_converted(unit_length='kpc')

            profile_kpc = profile_arcsec.new_profile_with_units_converted(unit_length='kpc', kpc_per_arcsec=2.0)

            with pytest.raises(exc.UnitsException):
                profile_kpc.new_profile_with_units_converted(unit_length='arcsec')

        def test__eps_to_counts_conversions_of_luminosity__conversions_convert_values(self):

            profile_eps = MockDimensionsProfile(
                position=(dim.Length(1.0, 'arcsec'), dim.Length(2.0, 'arcsec')),
                param_float=2.0,
                length=dim.Length(value=3.0, unit_length='arcsec'),
                luminosity=dim.Luminosity(value=4.0, unit_luminosity='eps'),
                mass=dim.Mass(value=5.0, unit_mass='angular'),
                mass_over_luminosity=dim.MassOverLuminosity(value=6.0, unit_luminosity='eps', unit_mass='angular'))

            assert profile_eps.position == (1.0, 2.0)
            assert profile_eps.position[0].unit_length == 'arcsec'
            assert profile_eps.position[1].unit_length == 'arcsec'
            assert profile_eps.param_float == 2.0
            assert profile_eps.length == 3.0
            assert profile_eps.length.unit_length == 'arcsec'
            assert profile_eps.luminosity == 4.0
            assert profile_eps.luminosity.unit_luminosity == 'eps'
            assert profile_eps.mass == 5.0
            assert profile_eps.mass.unit_mass == 'angular'
            assert profile_eps.mass_over_luminosity == 6.0
            assert profile_eps.mass_over_luminosity.unit == 'angular / eps'

            profile_eps = profile_eps.new_profile_with_units_converted(unit_luminosity='eps')

            assert profile_eps.position == (1.0, 2.0)
            assert profile_eps.position[0].unit_length == 'arcsec'
            assert profile_eps.position[1].unit_length == 'arcsec'
            assert profile_eps.param_float == 2.0
            assert profile_eps.length == 3.0
            assert profile_eps.length.unit_length == 'arcsec'
            assert profile_eps.luminosity == 4.0
            assert profile_eps.luminosity.unit_luminosity == 'eps'
            assert profile_eps.mass == 5.0
            assert profile_eps.mass.unit_mass == 'angular'
            assert profile_eps.mass_over_luminosity == 6.0
            assert profile_eps.mass_over_luminosity.unit == 'angular / eps'

            profile_counts = profile_eps.new_profile_with_units_converted(unit_luminosity='counts',
                                                                          exposure_time=10.0)

            assert profile_counts.position == (1.0, 2.0)
            assert profile_counts.position[0].unit_length == 'arcsec'
            assert profile_counts.position[1].unit_length == 'arcsec'
            assert profile_counts.param_float == 2.0
            assert profile_counts.length == 3.0
            assert profile_counts.length.unit_length == 'arcsec'
            assert profile_counts.luminosity == 40.0
            assert profile_counts.luminosity.unit_luminosity == 'counts'
            assert profile_counts.mass == 5.0
            assert profile_counts.mass.unit_mass == 'angular'
            assert profile_counts.mass_over_luminosity == pytest.approx(0.6, 1.0e-4)
            assert profile_counts.mass_over_luminosity.unit == 'angular / counts'

            profile_counts = profile_counts.new_profile_with_units_converted(unit_luminosity='counts')


            assert profile_counts.position == (1.0, 2.0)
            assert profile_counts.position[0].unit_length == 'arcsec'
            assert profile_counts.position[1].unit_length == 'arcsec'
            assert profile_counts.param_float == 2.0
            assert profile_counts.length == 3.0
            assert profile_counts.length.unit_length == 'arcsec'
            assert profile_counts.luminosity == 40.0
            assert profile_counts.luminosity.unit_luminosity == 'counts'
            assert profile_counts.mass == 5.0
            assert profile_counts.mass.unit_mass == 'angular'
            assert profile_counts.mass_over_luminosity == pytest.approx(0.6, 1.0e-4)
            assert profile_counts.mass_over_luminosity.unit == 'angular / counts'

            profile_eps = profile_counts.new_profile_with_units_converted(unit_luminosity='eps',
                                                                          exposure_time=10.0)

            assert profile_eps.position == (1.0, 2.0)
            assert profile_eps.position[0].unit_length == 'arcsec'
            assert profile_eps.position[1].unit_length == 'arcsec'
            assert profile_eps.param_float == 2.0
            assert profile_eps.length == 3.0
            assert profile_eps.length.unit_length == 'arcsec'
            assert profile_eps.luminosity == 4.0
            assert profile_eps.luminosity.unit_luminosity == 'eps'
            assert profile_eps.mass == 5.0
            assert profile_eps.mass.unit_mass == 'angular'
            assert profile_eps.mass_over_luminosity == pytest.approx(6.0, 1.0e-4)
            assert profile_eps.mass_over_luminosity.unit == 'angular / eps'

        def test__luminosity_conversion_requires_exposure_time_but_does_not_supply_it_raises_error(self):

            profile_eps = MockDimensionsProfile(
                position=(dim.Length(1.0, 'arcsec'), dim.Length(2.0, 'arcsec')),
                param_float=2.0,
                length=dim.Length(value=3.0, unit_length='arcsec'),
                luminosity=dim.Luminosity(value=4.0, unit_luminosity='eps'),
                mass=dim.Mass(value=5.0, unit_mass='angular'),
                mass_over_luminosity=dim.MassOverLuminosity(value=6.0, unit_luminosity='eps', unit_mass='angular'))

            with pytest.raises(exc.UnitsException):
                profile_eps.new_profile_with_units_converted(unit_luminosity='counts')

            profile_counts = profile_eps.new_profile_with_units_converted(unit_luminosity='counts', exposure_time=10.0)

            with pytest.raises(exc.UnitsException):
                profile_counts.new_profile_with_units_converted(unit_luminosity='eps')

        def test__angular_to_solMass_conversions_of_mass__conversions_convert_values(self):

            profile_angular = MockDimensionsProfile(
                position=(dim.Length(1.0, 'arcsec'), dim.Length(2.0, 'arcsec')),
                param_float=2.0,
                length=dim.Length(value=3.0, unit_length='arcsec'),
                luminosity=dim.Luminosity(value=4.0, unit_luminosity='eps'),
                mass=dim.Mass(value=5.0, unit_mass='angular'),
                mass_over_luminosity=dim.MassOverLuminosity(value=6.0, unit_luminosity='eps', unit_mass='angular'))

            assert profile_angular.position == (1.0, 2.0)
            assert profile_angular.position[0].unit_length == 'arcsec'
            assert profile_angular.position[1].unit_length == 'arcsec'
            assert profile_angular.param_float == 2.0
            assert profile_angular.length == 3.0
            assert profile_angular.length.unit_length == 'arcsec'
            assert profile_angular.luminosity == 4.0
            assert profile_angular.luminosity.unit_luminosity == 'eps'
            assert profile_angular.mass == 5.0
            assert profile_angular.mass.unit_mass == 'angular'
            assert profile_angular.mass_over_luminosity == 6.0
            assert profile_angular.mass_over_luminosity.unit == 'angular / eps'

            profile_angular = profile_angular.new_profile_with_units_converted(unit_mass='angular')

            assert profile_angular.position == (1.0, 2.0)
            assert profile_angular.position[0].unit_length == 'arcsec'
            assert profile_angular.position[1].unit_length == 'arcsec'
            assert profile_angular.param_float == 2.0
            assert profile_angular.length == 3.0
            assert profile_angular.length.unit_length == 'arcsec'
            assert profile_angular.luminosity == 4.0
            assert profile_angular.luminosity.unit_luminosity == 'eps'
            assert profile_angular.mass == 5.0
            assert profile_angular.mass.unit_mass == 'angular'
            assert profile_angular.mass_over_luminosity == 6.0
            assert profile_angular.mass_over_luminosity.unit == 'angular / eps'

            profile_solMass = profile_angular.new_profile_with_units_converted(unit_mass='solMass',
                                                                          critical_surface_density=10.0)

            assert profile_solMass.position == (1.0, 2.0)
            assert profile_solMass.position[0].unit_length == 'arcsec'
            assert profile_solMass.position[1].unit_length == 'arcsec'
            assert profile_solMass.param_float == 2.0
            assert profile_solMass.length == 3.0
            assert profile_solMass.length.unit_length == 'arcsec'
            assert profile_solMass.luminosity == 4.0
            assert profile_solMass.luminosity.unit_luminosity == 'eps'
            assert profile_solMass.mass == 50.0
            assert profile_solMass.mass.unit_mass == 'solMass'
            assert profile_solMass.mass_over_luminosity == pytest.approx(60.0, 1.0e-4)
            assert profile_solMass.mass_over_luminosity.unit == 'solMass / eps'

            profile_solMass = profile_solMass.new_profile_with_units_converted(unit_mass='solMass')

            assert profile_solMass.position == (1.0, 2.0)
            assert profile_solMass.position[0].unit_length == 'arcsec'
            assert profile_solMass.position[1].unit_length == 'arcsec'
            assert profile_solMass.param_float == 2.0
            assert profile_solMass.length == 3.0
            assert profile_solMass.length.unit_length == 'arcsec'
            assert profile_solMass.luminosity == 4.0
            assert profile_solMass.luminosity.unit_luminosity == 'eps'
            assert profile_solMass.mass == 50.0
            assert profile_solMass.mass.unit_mass == 'solMass'
            assert profile_solMass.mass_over_luminosity == pytest.approx(60.0, 1.0e-4)
            assert profile_solMass.mass_over_luminosity.unit == 'solMass / eps'

            profile_angular = profile_solMass.new_profile_with_units_converted(unit_mass='angular',
                                                                          critical_surface_density=10.0)

            assert profile_angular.position == (1.0, 2.0)
            assert profile_angular.position[0].unit_length == 'arcsec'
            assert profile_angular.position[1].unit_length == 'arcsec'
            assert profile_angular.param_float == 2.0
            assert profile_angular.length == 3.0
            assert profile_angular.length.unit_length == 'arcsec'
            assert profile_angular.luminosity == 4.0
            assert profile_angular.luminosity.unit_luminosity == 'eps'
            assert profile_angular.mass == 5.0
            assert profile_angular.mass.unit_mass == 'angular'
            assert profile_angular.mass_over_luminosity == pytest.approx(6.0, 1.0e-4)
            assert profile_angular.mass_over_luminosity.unit == 'angular / eps'

        def test__mass_conversion_requires_critical_surface_density_but_does_not_supply_it_raises_error(self):

            profile_angular = MockDimensionsProfile(
                position=(dim.Length(1.0, 'arcsec'), dim.Length(2.0, 'arcsec')),
                param_float=2.0,
                length=dim.Length(value=3.0, unit_length='arcsec'),
                luminosity=dim.Luminosity(value=4.0, unit_luminosity='eps'),
                mass=dim.Mass(value=5.0, unit_mass='angular'),
                mass_over_luminosity=dim.MassOverLuminosity(value=6.0, unit_luminosity='eps', unit_mass='angular'))

            with pytest.raises(exc.UnitsException):
                profile_angular.new_profile_with_units_converted(unit_mass='solMass')

            profile_solMass = profile_angular.new_profile_with_units_converted(unit_mass='solMass', critical_surface_density=10.0)

            with pytest.raises(exc.UnitsException):
                profile_solMass.new_profile_with_units_converted(unit_mass='angular')


class TestUnitCheckConversionWrapper(object):

    def test__profile_length_units_calculations__profile_is_converted_for_calculation_if_different_to_input_units(self):

        profile = MockDimensionsProfile(length=dim.Length(3.0, 'arcsec'))

        cosmo = MockCosmology(kpc_per_arcsec=2.0)

        # length: arcsec -> arcsec, stays 3.0,  length_input: arcsec -> arcsec, stays 1.0

        length_input = dim.Length(1.0, 'arcsec')
        length = profile.unit_length_calc(length_input=length_input, unit_length='arcsec')
        assert length.unit_length == 'arcsec'
        assert length == 4.0

        # length: arcsec -> arcsec, stays 3.0,  length_input  kpc -> arcsec, converts to 1.0 / 2.0 = 0.5

        length_input = dim.Length(1.0, 'kpc')
        length = profile.unit_length_calc(length_input=length_input, unit_length='arcsec',
                                          redshift_profile=0.5, cosmology=cosmo)
        assert length.unit_length == 'arcsec'
        assert length == 3.5

        # length: arcsec -> kpc, converts to 3.0 * 2.0 = 6.0,  length_input  kpc -> kpc, stays 1.0

        length_input = dim.Length(1.0, 'kpc')
        length = profile.unit_length_calc(length_input=length_input, unit_length='kpc',
                                          redshift_profile=0.5, cosmology=cosmo)
        assert length.unit_length == 'kpc'
        assert length == 7.0
        
        profile = MockDimensionsProfile(length=dim.Length(3.0, 'kpc'))

        # length: kpc -> kpc, stays 3.0,  length_input: kpc -> kpc, stays 1.0

        length_input = dim.Length(1.0, 'kpc')
        length = profile.unit_length_calc(length_input=length_input, unit_length='kpc',
                                          redshift_profile=0.5, cosmology=cosmo)
        assert length.unit_length == 'kpc'
        assert length == 4.0

        # length: kpc -> kpc, stays 3.0,  length_input: arcsec -> kpc, convert to 1.0 * 2.0 = 2.0

        length_input = dim.Length(1.0, 'arcsec')
        length = profile.unit_length_calc(length_input=length_input, unit_length='kpc',
                                          redshift_profile=0.5, cosmology=cosmo)
        assert length.unit_length == 'kpc'
        assert length == 5.0

        # length: kpc -> arcsec, converts to 3.0 / 2.0 = 1.5,  length_input: kpc -> kpc, stays 1.0

        length_input = dim.Length(1.0, 'arcsec')
        length = profile.unit_length_calc(length_input=length_input, unit_length='arcsec',
                                          redshift_profile=0.5, cosmology=cosmo)
        assert length.unit_length == 'arcsec'
        assert length == 2.5

    def test__profile_luminosity_units_calculations__profile_is_converted_for_calculation_if_different_to_input_units(self):
        
        profile = MockDimensionsProfile(luminosity=dim.Luminosity(3.0, 'eps'))

        cosmo = MockCosmology()

        # luminosity: eps -> eps, stays 3.0,  luminosity_input: eps -> eps, stays 1.0

        luminosity_input = dim.Luminosity(1.0, 'eps')
        luminosity = profile.unit_luminosity_calc(luminosity_input=luminosity_input, unit_luminosity='eps')
        assert luminosity.unit_luminosity == 'eps'
        assert luminosity == 4.0

        # luminosity: eps -> eps, stays 3.0,  luminosity_input  counts -> eps, converts to 1.0 / 2.0 = 0.5

        luminosity_input = dim.Luminosity(1.0, 'counts')
        luminosity = profile.unit_luminosity_calc(luminosity_input=luminosity_input, unit_luminosity='eps',
                                                  redshift_profile=0.5, cosmology=cosmo, exposure_time=2.0)
        assert luminosity.unit_luminosity == 'eps'
        assert luminosity == 3.5

        # luminosity: eps -> counts, converts to 3.0 * 2.0 = 6.0,  luminosity_input  counts -> counts, stays 1.0

        luminosity_input = dim.Luminosity(1.0, 'counts')
        luminosity = profile.unit_luminosity_calc(luminosity_input=luminosity_input, unit_luminosity='counts',
                                          redshift_profile=0.5, cosmology=cosmo, exposure_time=2.0)
        assert luminosity.unit_luminosity == 'counts'
        assert luminosity == 7.0

        profile = MockDimensionsProfile(luminosity=dim.Luminosity(3.0, 'counts'))

        # luminosity: counts -> counts, stays 3.0,  luminosity_input: counts -> counts, stays 1.0

        luminosity_input = dim.Luminosity(1.0, 'counts')
        luminosity = profile.unit_luminosity_calc(luminosity_input=luminosity_input, unit_luminosity='counts',
                                          redshift_profile=0.5, cosmology=cosmo, exposure_time=2.0)
        assert luminosity.unit_luminosity == 'counts'
        assert luminosity == 4.0

        # luminosity: counts -> counts, stays 3.0,  luminosity_input: eps -> counts, convert to 1.0 * 2.0 = 2.0

        luminosity_input = dim.Luminosity(1.0, 'eps')
        luminosity = profile.unit_luminosity_calc(luminosity_input=luminosity_input, unit_luminosity='counts',
                                          redshift_profile=0.5, cosmology=cosmo, exposure_time=2.0)
        assert luminosity.unit_luminosity == 'counts'
        assert luminosity == 5.0

        # luminosity: counts -> eps, converts to 3.0 / 2.0 = 1.5,  luminosity_input: counts -> counts, stays 1.0

        luminosity_input = dim.Luminosity(1.0, 'eps')
        luminosity = profile.unit_luminosity_calc(luminosity_input=luminosity_input, unit_luminosity='eps',
                                          redshift_profile=0.5, cosmology=cosmo, exposure_time=2.0)
        assert luminosity.unit_luminosity == 'eps'
        assert luminosity == 2.5

    def test__profile_mass_units_calculations__profile_is_converted_for_calculation_if_different_to_input_units(self):

        profile = MockDimensionsProfile(length=dim.Length(1.0, 'arcsec'), mass=dim.Mass(3.0, 'angular'))

        cosmo = MockCosmology(critical_surface_density=2.0)

        # mass: angular -> angular, stays 3.0,  mass_input: angular -> angular, stays 1.0

        mass_input = dim.Mass(1.0, 'angular')
        mass = profile.unit_mass_calc(mass_input=mass_input, unit_mass='angular',
                                      redshift_profile=0.5, redshift_source=1.0, cosmology=cosmo)
        assert mass.unit_mass == 'angular'
        assert mass == 4.0

        # mass: angular -> angular, stays 3.0,  mass_input  solMass -> angular, stays as 1.0

        mass_input = dim.Mass(1.0, 'solMass')
        mass = profile.unit_mass_calc(mass_input=mass_input, unit_mass='angular',
                                      redshift_profile=0.5, redshift_source=1.0, cosmology=cosmo)
        assert mass.unit_mass == 'angular'
        assert mass == 4.0

        # mass: angular -> solMass, converts to 3.0 * 2.0 = 6.0,  mass_input  solMass -> solMass, stays 1.0

        mass_input = dim.Mass(1.0, 'solMass')
        mass = profile.unit_mass_calc(mass_input=mass_input, unit_mass='solMass',
                                          redshift_profile=0.5, redshift_source=1.0, cosmology=cosmo)
        assert mass.unit_mass == 'solMass'
        assert mass == 7.0

        profile = MockDimensionsProfile(length=dim.Length(1.0, 'arcsec'), mass=dim.Mass(3.0, 'solMass'))

        # mass: solMass -> solMass, stays 3.0,  mass_input: solMass -> solMass, stays 1.0

        mass_input = dim.Mass(1.0, 'solMass')
        mass = profile.unit_mass_calc(mass_input=mass_input, unit_mass='solMass',
                                          redshift_profile=0.5, redshift_source=1.0, cosmology=cosmo)
        assert mass.unit_mass == 'solMass'
        assert mass == 4.0

        # mass: solMass -> solMass, stays 3.0,  mass_input: angular -> solMass, convert to 1.0 * 2.0 = 2.0

        mass_input = dim.Mass(1.0, 'angular')
        mass = profile.unit_mass_calc(mass_input=mass_input, unit_mass='solMass',
                                          redshift_profile=0.5, redshift_source=1.0, cosmology=cosmo)
        assert mass.unit_mass == 'solMass'
        assert mass == 5.0

        # mass: solMass -> angular, stays 3.0,  mass_input: solMass -> solMass, stays 1.0

        mass_input = dim.Mass(1.0, 'angular')
        mass = profile.unit_mass_calc(mass_input=mass_input, unit_mass='angular',
                                          redshift_profile=0.5, redshift_source=1.0, cosmology=cosmo)
        assert mass.unit_mass == 'angular'
        assert mass == 4.0