# qfm 

This repository uses the `simsopt` python package for Stellarator analysis and optimisation. Our scripts can be used for the following:

1. Create Poincare field line tracing data and plots.
2. Create an initial QFM surface and plot.
3. Create a range of QFM surfaces by loading an initial QFM surface and iteratively optimising with new toroidal flux constraints.
4. Increase the Fourier mode resolution of QFM surfaces.
5. Create Boozer surfaces from QFM surfaces and extract and iota profile.
6. Make a coordinate map from QFM to Boozer coordinates.
7. Extract the $\mathbf{b \cdot \hat{n}}$ coefficient spectrum on a rational surface.



## Information for users

These scripts should be run in the `simsopt` python environment. 

Here is the link to download `simsopt`:https://pypi.org/project/simsopt/.

Here is a link to documentation for `simsopt`: https://simsopt.readthedocs.io/latest/.

Any comments/suggestions/queries can be sent to: Louis Day <louisday81@gmail.com>.

This work was completed during a summer internship at the Max Planck Institute for Plasma Physics in Greifswald, Germany.
There is an accompanying report on this project.
## Scripts
The following scripts are available:
1. `w7x_config.py` defines magnetic configurations for Wendelstein 7-X by specifying coil currents.
2. `poincare_fieldlines.py` traces field lines for a given W7-X configuration. Plots cross-sections and saves data.
3. `initial_qfm_surface.py` makes a QFM surface with a given toroidal flux. Plots cross-sections with
field line data and saves.
4. `qfm_full_set.py` makes a full set of QFM surfaces labelled by toroidal flux. Plots cross-sections, saves plots
and surface data.
5. `mpol_increase.py` increases the number of available Fourier modes for a given set of QFM surfaces. Sequentially
 increases $\textrm{mpol=ntor}$ and saves each QFM surface. Plots cross-sections and saves.
6. `boozer_from_qfm.py` makes Boozer surfaces from a given set of QFM surfaces. Extracts the iota profile and plot. Plots
 cross-sections and saves surfaces/plots/magnetic shear.
7. `boozer_to_qfm_spline.py` learns a coordinate map $(\theta_{Q},\phi_{Q}) \to (\theta_B,\phi_B)$ from QFM
 to Boozer coordinates for a given toroidal flux (where good QFM and Boozer surfaces exist). Apply this map to the 
rational QFM surface (which does not have a converged Boozer surface) to achieve an approximation to a 
Boozer-parametrised rational surface.
8. `bn_spectrum` calculate and Fourier decompose the normalised 
$b_r = \mathbf{b \cdot \hat{n}} = \mathbf{B \cdot \hat{n}}/B_\phi$.

To Fourier decompose this quantity consider the 
Fourier series:

$$b_r(\theta, \phi) = \sum_{m,n} A_{m,n} \sin(m\theta-n\phi) + B_{m,n}\cos(m\theta - n\phi).$$

We can make use of the following orthogonality arguments

$$ \langle \sin(m\theta - n\phi), \sin(\mu \theta-\nu\phi)\rangle = \frac{1}{2}\delta_{m\mu} \delta_{n\nu}, $$
$$\langle \cos(m\theta - n\phi), \cos(\mu \theta-\nu\phi)\rangle = \frac{1}{2}\delta_{m\mu} \delta_{n\nu}, $$
$$ \langle\sin(m\theta-n\phi),\cos(\mu \theta-\nu\phi)\rangle =0. $$

Using the above we can write

$$ A_{m,n} = 2\langle b_r, \sin(m\theta-n\phi)\rangle, $$
$$ B_{m,n} = 2\langle b_r, \cos(m\theta-n\phi)\rangle. $$

Where

```math
\langle f,g \rangle =
\frac{n_{fp}}{4\pi^2}
\int_{0}^{2\pi/n_{fp}} \mathrm{d}\phi
\int_{0}^{2\pi} \mathrm{d}\theta\,
f(\theta,\phi)\cdot g(\theta,\phi),
```

defines an inner product. In `bn_spectrum.py` this integral is approximated using a weighted summation.
