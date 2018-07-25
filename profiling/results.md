# Deflection

Running 100 calls on mass_profiles.EllipticalIsothermal(axis_ratio=0.9)

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
old_method: 1.2667009830474854

new_method: 0.026113033294677734

x faster: 48.50838157149901

#### EllipticalIsothermal
old_method: 1.4990592002868652

new_method: 0.020721912384033203

x faster: 72.34174011091423

#### SphericalIsothermal
old_method: 1.1057920455932617

new_method: 0.01303863525390625

x faster: 84.8088794616735

#### SphericalCoredPowerLaw
old_method: 1.3261933326721191

new_method: 0.028895854949951172

x faster: 45.895625340352154

#### SphericalCoredIsothermal
old_method: 1.339472770690918

new_method: 0.029388904571533203

x faster: 45.577499067058234

#### ExternalShear
old_method: 1.430459976196289

new_method: 0.018290281295776367

x faster: 78.20874665971452


# Intensity

#### To Eccentric Radius

Running 100 calls on EllipticalProfile.to_eccentric_radius.

classic_grid_to_eccentric_radius: 0.9260859489440918
new_grid_to_eccentric_radius:   0.011136054992675781

## Results from Several Light Profiles

#### EllipticalSersic
old_method: 1.1218419075012207

new_method: 0.012676715850830078

x faster: 88.49625728794433

#### EllipticalExponential
old_method: 1.0966298580169678

new_method: 0.012710094451904297

x faster: 86.28022885012193

#### EllipticalDevVaucouleurs
old_method: 1.104621171951294

new_method: 0.012662172317504883

x faster: 87.23788811689168

#### EllipticalCoreSersic
old_method: 1.2265520095825195

new_method: 0.017539024353027344

x faster: 69.93273883964983


# Analysis

Before changes: 34.58575987815857

After changes: 0.5687899589538574

60.81 x faster!

# Misc

#### map_data_sub_to_image

100x analysis:
5.609487056732178 -> 2.8315680027008057

By using numpy method rather than loop

about half