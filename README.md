<h1 id="pyautolens">PyAutoLens</h1>
<p>PyAutoLens makes it simple to model strong gravitational lenses. It is based on the following papers:</p>
<p>https://arxiv.org/abs/1412.7436<br/> https://arxiv.org/abs/1708.07377</p>
<h2 id="contact">Contact</h2>
<p>Before using PyAutoLens, I recommend you contact us on our <a href="https://pyautolens.slack.com/">SLACK channel</a>. Here, I can give you the latest updates on installation, functionality and the best way to use PyAutoLens for your science case.</p>
<p>Unfortunately, SLACK is invitation-only, so first send me an <a href="https://github.com/Jammy2211">email</a> requesting an invite.</p>
<h2 id="installation">Installation</h2>
<p>AutoLens requires <a href="http://johannesbuchner.github.io/pymultinest-tutorial/install.html">PyMultiNest</a> and <a href="https://github.com/numba/numba">Numba</a>.</p>
<pre><code>$ pip install autolens</code></pre>
<p>Known issues with the installation can be found in the file <a href="https://github.com/Jammy2211/PyAutoLens/blob/master/INSTALL.notes">INSTALL.notes</a></p>
<h2 id="python-example">Python Example</h2>
<p>With PyAutoLens, you can begin modeling a lens in just a couple of minutes. The example below demonstrates a simple analysis which fits a lens galaxy's light, mass and a source galaxy.</p>
<pre class="sourceCode python"><code class="sourceCode python"><span class="ch">from</span> autolens.pipeline <span class="ch">import</span> phase <span class="ch">as</span> ph
<span class="ch">from</span> autolens.autofit <span class="ch">import</span> non_linear <span class="ch">as</span> nl
<span class="ch">from</span> autolens.lensing <span class="ch">import</span> galaxy_prior <span class="ch">as</span> gp
<span class="ch">from</span> autolens.imaging <span class="ch">import</span> image <span class="ch">as</span> im
<span class="ch">from</span> autolens.profiles <span class="ch">import</span> light_profiles <span class="ch">as</span> lp
<span class="ch">from</span> autolens.profiles <span class="ch">import</span> mass_profiles <span class="ch">as</span> mp
<span class="ch">from</span> autolens.plotting <span class="ch">import</span> fitting_plotters
<span class="ch">import</span> os

<span class="co"># In this example, we&#39;ll generate a phase which fits a lens + source plane system.</span>

<span class="co"># First, lets setup the path to this script so we can easily load the example data.</span>
path = <span class="st">&quot;{}&quot;</span>.<span class="dt">format</span>(os.path.dirname(os.path.realpath(<span class="ot">__file__</span>)))

<span class="co"># Now, load the image, noise-map and PSF from the &#39;data&#39; folder.</span>
image = im.load_imaging_from_path(image_path=path + <span class="st">&#39;/data/image.fits&#39;</span>,
                                  noise_map_path=path + <span class="st">&#39;/data/noise_map.fits&#39;</span>,
                                  psf_path=path + <span class="st">&#39;/data/psf.fits&#39;</span>, pixel_scale=<span class="fl">0.1</span>)

<span class="co"># We&#39;re going to model our lens galaxy using a light profile (an elliptical Sersic) and mass profile</span>
<span class="co"># (a singular isothermal sphere). We load these profiles from the &#39;light_profile (lp)&#39; and &#39;mass_profile (mp)&#39;</span>
<span class="co"># modules (check out the source code to see all the profiles that are available).</span>

<span class="co"># To setup our model galaxies, we use the &#39;galaxy_model&#39; module and GalaxyModel class. </span>
<span class="co"># A GalaxyModel represents a galaxy where the parameters of its associated profiles are </span>
<span class="co"># variable and fitted for by the analysis.</span>
lens_galaxy_model = gp.GalaxyModel(light=lp.AbstractEllipticalSersic, mass=mp.EllipticalIsothermal)
source_galaxy_model = gp.GalaxyModel(light=lp.AbstractEllipticalSersic)

<span class="co"># To perform the analysis, we set up a phase using the &#39;phase&#39; module (imported as &#39;ph&#39;).</span>
<span class="co"># A phase takes our galaxy models and fits their parameters using a non-linear search (in this case, MultiNest).</span>
phase = ph.LensSourcePlanePhase(lens_galaxies=[lens_galaxy_model], source_galaxies=[source_galaxy_model],
                                optimizer_class=nl.MultiNest, phase_name=<span class="st">&#39;phase_example&#39;</span>)

<span class="co"># We run the phase on the image, print the results and plot the fit.</span>
results = phase.run(image)
<span class="dt">print</span>(results)
fitting_plotters.plot_fitting_subplot(fit=results.fit)</code></pre>
<h2 id="advanced-lens-modeling">Advanced Lens Modeling</h2>
<p>The example above shows the simplest analysis one can perform in PyAutoLens. PyAutoLens's advanced modeling features include:</p>
<ul>
<li><strong>Pipelines</strong> - build automated analysis pipelines to fit complex lens models to large samples of strong lenses.</li>
<li><strong>Inversions</strong> - Reconstruct complex source galaxy morphologies on a variety of pixel-grids.</li>
<li><strong>Adaption</strong> - (October 2018) - Adapt the lensing analysis to the features of the observed strong lens imaging.</li>
<li><strong>Multi-Plane</strong> - (November 2018) Model multi-plane lenses, including systems with multiple lensed source galaxies.</li>
</ul>
<h2 id="howtolens">HowToLens</h2>
<p>Detailed tutorials demonstrating how to use PyAutoLens can be found in the 'howtolens' folder:</p>
<ul>
<li><strong>Introduction</strong> - How to use PyAutolens, familiarizing you with the interface and project structure.</li>
<li><strong>Lens Modeling</strong> - How to model strong lenses, including a primer on Bayesian non-linear analysis.</li>
<li><strong>Pipelines</strong> - How to build pipelines and tailor them to your own science case.</li>
<li><strong>Inversions</strong> - How to perform pixelized reconstructions of the source-galaxy.</li>
</ul>
<h2 id="support-discussion">Support &amp; Discussion</h2>
<p>If you're having difficulty with installation, lens modeling, or just want a chat, feel free to message us on our <a href="https://pyautolens.slack.com/">SLACK channel</a>.</p>
<h2 id="contributing">Contributing</h2>
<p>If you have any suggestions or would like to contribute please get in touch.</p>
