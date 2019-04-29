import pytest

from autolens import exc
from autofit.tools.dimension_type import map_types
from autolens.model import dimensions as dim


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

        unit_angular = dim.Mass(value=2.0)

        assert unit_angular == 2.0
        assert unit_angular.unit_mass == 'angular'

        unit_angular = unit_angular.convert(unit_mass='angular')

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular'

        unit_sol_mass = unit_angular.convert(unit_mass='solMass', critical_surface_density=2.0)

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == 'solMass'

        unit_sol_mass = unit_sol_mass.convert(unit_mass='solMass')

        assert unit_sol_mass == 4.0
        assert unit_sol_mass.unit == 'solMass'

        unit_angular = unit_sol_mass.convert(unit_mass='angular', critical_surface_density=2.0)

        assert unit_angular == 2.0
        assert unit_angular.unit == 'angular'

        with pytest.raises(exc.UnitsException):
            unit_angular.convert(unit_mass='solMass')
            unit_sol_mass.convert(unit_mass='angular')
            unit_angular.convert(unit_mass='lol')


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

    @map_types
    def __init__(self,
                 position: dim.Position = (1.0, 2.0),
                 param_float: float = 2.0,
                 luminosity: dim.Luminosity = 3.0,
                 length: dim.Length = 4.0,
                 mass_over_luminosity: dim.MassOverLuminosity = 5.0):

        super(MockDimensionsProfile, self).__init__()

        self.position = position
        self.param_float = param_float
        self.luminosity = luminosity
        self.length = length
        self.mass_over_luminosity = mass_over_luminosity


class TestDimensionsProfile(object):

    def test__arcsec_to_kpc_conversions_of_length__float_and_tuple_length__conversion_converts_values(self):

        profile_arcsec = MockDimensionsProfile(position=(1.0, 2.0),
                                               param_float=2.0,
                                               luminosity=3.0,
                                               length=4.0,
                                               mass_over_luminosity=5.0)

        assert profile_arcsec.position == (1.0, 2.0)
        assert profile_arcsec.position[0].unit_length == 'arcsec'
        assert profile_arcsec.position[1].unit_length == 'arcsec'
        assert profile_arcsec.length == 4.0
        assert profile_arcsec.length.unit_length == 'arcsec'

        assert profile_arcsec.param_float == 2.0
        assert profile_arcsec.luminosity == 3.0
        assert profile_arcsec.luminosity.unit_luminosity == 'eps'
        assert profile_arcsec.mass_over_luminosity == 5.0
        assert profile_arcsec.mass_over_luminosity.unit == 'angular / eps'

        profile_arcsec = profile_arcsec.new_profile_with_units_converted(unit_length='arcsec')

        assert profile_arcsec.position == (1.0, 2.0)
        assert profile_arcsec.position[0].unit == 'arcsec'
        assert profile_arcsec.position[1].unit == 'arcsec'
        assert profile_arcsec.length == 4.0
        assert profile_arcsec.length.unit == 'arcsec'

        assert profile_arcsec.param_float == 2.0
        assert profile_arcsec.luminosity == 3.0
        assert profile_arcsec.luminosity.unit == 'eps'
        assert profile_arcsec.mass_over_luminosity == 5.0
        assert profile_arcsec.mass_over_luminosity.unit == 'angular / eps'

        profile_kpc = profile_arcsec.new_profile_with_units_converted(unit_length='kpc', kpc_per_arcsec=2.0)

        assert profile_kpc.position == (2.0, 4.0)
        assert profile_kpc.position[0].unit == 'kpc'
        assert profile_kpc.position[1].unit == 'kpc'
        assert profile_kpc.length == 8.0
        assert profile_kpc.length.unit == 'kpc'

        assert profile_kpc.param_float == 2.0
        assert profile_kpc.luminosity == 3.0
        assert profile_kpc.luminosity.unit == 'eps'
        assert profile_kpc.mass_over_luminosity == 5.0
        assert profile_kpc.mass_over_luminosity.unit == 'angular / eps'

        profile_kpc = profile_kpc.new_profile_with_units_converted(unit_length='kpc')

        assert profile_kpc.position == (2.0, 4.0)
        assert profile_kpc.position[0].unit == 'kpc'
        assert profile_kpc.position[1].unit == 'kpc'
        assert profile_kpc.length == 8.0
        assert profile_kpc.length.unit == 'kpc'

        assert profile_kpc.param_float == 2.0
        assert profile_kpc.luminosity == 3.0
        assert profile_kpc.luminosity.unit == 'eps'
        assert profile_kpc.mass_over_luminosity == 5.0
        assert profile_kpc.mass_over_luminosity.unit == 'angular / eps'

        profile_arcsec = profile_kpc.new_profile_with_units_converted(unit_length='arcsec', kpc_per_arcsec=2.0)

        assert profile_arcsec.position == (1.0, 2.0)
        assert profile_arcsec.position[0].unit == 'arcsec'
        assert profile_arcsec.position[1].unit == 'arcsec'
        assert profile_arcsec.length == 4.0
        assert profile_arcsec.length.unit == 'arcsec'

        assert profile_arcsec.param_float == 2.0
        assert profile_arcsec.luminosity == 3.0
        assert profile_arcsec.luminosity.unit == 'eps'
        assert profile_arcsec.mass_over_luminosity == 5.0
        assert profile_arcsec.mass_over_luminosity.unit == 'angular / eps'

    def test__conversion_requires_kpc_per_arcsec_but_does_not_supply_it_raises_error(self):

        profile_arcsec = MockDimensionsProfile(position=(1.0, 1.0))

        with pytest.raises(exc.UnitsException):
            profile_arcsec.new_profile_with_units_converted(unit_length='kpc')

        profile_kpc = profile_arcsec.new_profile_with_units_converted(unit_length='kpc', kpc_per_arcsec=2.0)

        with pytest.raises(exc.UnitsException):
            profile_kpc.new_profile_with_units_converted(unit_length='arcsec')

    def test__eps_to_counts_conversions_of_luminosity__conversions_convert_values(self):

        profile_eps = MockDimensionsProfile(position=(1.0, 2.0),
                                            param_float=2.0,
                                            luminosity=3.0,
                                            length=4.0,
                                            mass_over_luminosity=5.0)

        profile_eps = profile_eps.new_profile_with_units_converted(unit_luminosity='eps')

        assert profile_eps.luminosity == 3.0
        assert profile_eps.luminosity.unit == 'eps'
        assert profile_eps.mass_over_luminosity == 5.0
        assert profile_eps.mass_over_luminosity.unit == 'angular / eps'

        assert profile_eps.position == (1.0, 2.0)
        assert profile_eps.position[0].unit == 'arcsec'
        assert profile_eps.position[1].unit == 'arcsec'
        assert profile_eps.length == 4.0
        assert profile_eps.length.unit == 'arcsec'
        assert profile_eps.param_float == 2.0

        profile_eps = profile_eps.new_profile_with_units_converted(unit_luminosity='eps')

        assert profile_eps.luminosity == 3.0
        assert profile_eps.luminosity.unit == 'eps'
        assert profile_eps.mass_over_luminosity == 5.0
        assert profile_eps.mass_over_luminosity.unit == 'angular / eps'

        assert profile_eps.position == (1.0, 2.0)
        assert profile_eps.position[0].unit == 'arcsec'
        assert profile_eps.position[1].unit == 'arcsec'
        assert profile_eps.length == 4.0
        assert profile_eps.length.unit == 'arcsec'
        assert profile_eps.param_float == 2.0

        profile_counts = profile_eps.new_profile_with_units_converted(unit_luminosity='counts',
                                                                      exposure_time=10.0)

        assert profile_counts.luminosity == 30.0
        assert profile_counts.luminosity.unit == 'counts'
        assert profile_counts.mass_over_luminosity == 0.5
        assert profile_counts.mass_over_luminosity.unit == 'angular / counts'

        assert profile_counts.position == (1.0, 2.0)
        assert profile_counts.position[0].unit == 'arcsec'
        assert profile_counts.position[1].unit == 'arcsec'
        assert profile_counts.length == 4.0
        assert profile_counts.length.unit == 'arcsec'
        assert profile_counts.param_float == 2.0

        profile_counts = profile_counts.new_profile_with_units_converted(unit_luminosity='counts')

        assert profile_counts.luminosity == 30.0
        assert profile_counts.luminosity.unit == 'counts'
        assert profile_counts.mass_over_luminosity == 0.5
        assert profile_counts.mass_over_luminosity.unit == 'angular / counts'

        assert profile_counts.position == (1.0, 2.0)
        assert profile_counts.position[0].unit == 'arcsec'
        assert profile_counts.position[1].unit == 'arcsec'
        assert profile_counts.length == 4.0
        assert profile_counts.length.unit == 'arcsec'
        assert profile_counts.param_float == 2.0

        profile_eps = profile_counts.new_profile_with_units_converted(unit_luminosity='eps',
                                                                      exposure_time=10.0)

        assert profile_eps.luminosity == 3.0
        assert profile_eps.luminosity.unit == 'eps'
        assert profile_eps.mass_over_luminosity == 5.0
        assert profile_eps.mass_over_luminosity.unit == 'angular / eps'

        assert profile_eps.position == (1.0, 2.0)
        assert profile_eps.position[0].unit == 'arcsec'
        assert profile_eps.position[1].unit == 'arcsec'
        assert profile_eps.length == 4.0
        assert profile_eps.length.unit == 'arcsec'
        assert profile_eps.param_float == 2.0

    def test__luminosity_conversion_requires_exposure_time_but_does_not_supply_it_raises_error(self):

        profile_eps = MockDimensionsProfile(position=(1.0, 2.0), length=4.0, param_float=2.0, luminosity=3.0,
                                            mass_over_luminosity=5.0)

        with pytest.raises(exc.UnitsException):
            profile_eps.new_profile_with_units_converted(unit_luminosity='counts')

        profile_counts = profile_eps.new_profile_with_units_converted(unit_luminosity='counts', exposure_time=10.0)

        with pytest.raises(exc.UnitsException):
            profile_counts.new_profile_with_units_converted(unit_luminosity='eps')