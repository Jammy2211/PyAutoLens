# Deflection

Running 100 tests on mass_profiles.EllipticalIsothermal(axis_ratio=0.9)

#### Original Coordinate Transformation
looping numpy array and applying function to each coordinate pair

0.9136519432067871

#### New Coordinate Transformation

applying operations to a numpy array

0.0074498653411865234


#### Transforming and Returning to Original

current_transform_and_back: 1.3987503051757812
new_transform_and_back: 0.0332338809967041

#### Overall Result

Transforming, calculating deflection and transforming back. 65x speed increase.

current_deflection_elliptical_isothermal: 1.5677199363708496
new_deflection_elliptical_isothermal: 0.023957014083862305

## Results from Several Mass Profiles

#### SphericalPowerLaw
old_method: 1.4004719257354736
new_method: 0.029329299926757812
x faster: 47.7499268387852

#### EllipticalIsothermal
old_method: 1.4174158573150635
new_method: 0.02125692367553711
x faster: 66.6801969537226

#### SphericalIsothermal
old_method: 1.436042070388794
new_method: 0.015048027038574219
x faster: 95.43058812345522

#### SphericalCoredPowerLaw
old_method: 1.4370160102844238
new_method: 0.029826879501342773
x faster: 48.178556869139825

#### SphericalCoredIsothermal
old_method: 1.3838720321655273  
new_method: 0.03012371063232422
x faster: 45.93962706176592

#### ExternalShear
old_method: 1.3890931606292725
new_method: 0.016950130462646484
x faster: 81.95176808169465