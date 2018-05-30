### Mask Convention
- We need to ensure that it's the same as NumPy

### Boundary Convolution
- We will adapt the frame convolver to account for masked pixels.  
- No conclusion on buffer simulation

### Matrix/Graph Representation  
- We will use NumPy matrices for ease.  

### RayTracing  
- Galaxies will be independent objects with intrinsic redshift.   

### Pipeline/Analysis classes  
- RayTracing accepts image_plane\_grid, lens\_galaxies, source\_galaxies Maybe subclassed for pixelisation?   
- Analysis class accepts lens\_galaxy\_priors, source\_galaxy\_prior and image
- Analysis class computes grids etc. from image
- Analysis has compute_liklihood function
- GalaxyPrior class wraps mass and light profile priors
- Analysis uses lists of galaxy priors in conjunction with multinest to construct RayTracing instances to pass to compute_liklihood function
- Pass list of analysis instances into Pipeline

### Pipeline linking
- Generically link profiles by geometry, einstein mass and luminosity
- Link galaxies/profiles by ordering within analysis instances
- Hyperparameter optimisation between analyses
- Hyperparameter optimisation phase is implicit because it depends on chosen model classes
- May want decision tree pipelines that compare multiple models in a single analysis
