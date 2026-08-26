"""Make a QFM surface with a given toroidal flux.

We use simsopt's QfmSurface object with the optimisers to compute a QFM surface starting from a circular tube centred
around the magnetic axis.

Key parameters to set:
- the configuration to use.
- the flux to target. We recommend targeting fluxes that correspond to surfaces inside the nested surfaces region.
ie. in the range ~0.2 to 1.5Wb for the standard configuration of W7-X.
- the Fourier modes to optimise (mpol=3 is the safest setting for the initial surfaces).
- the grid to optimise on. We find that NPHI = NTHETA = 40 is sufficient.
- the grid to plot on. This takes the surface and samples points to make a plot and so can be set much higher than above.

The script then makes a QFM surface, saves it and plots at phi=0:
 - the Poincare scatter data
 - the magnetic axis
 - the initial tube surface cross-section
 - the QFM surface cross-section


To compute QFM surfaces outside the recommended flux range it is safer/more successful to use the
flux continuation script.


"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import brentq

from simsopt.field import BiotSavart
from simsopt.geo import QfmSurface, SurfaceRZFourier, ToroidalFlux

from w7x_config import NFP, build_field
from poincare_fieldlines import PHIS_OVER_PI as TRACED_PHIS

CONFIG = "standard"
SAVE = True             # True means save the surface. False means do not save the surface but still show the plot.

TARGET_FLUX = 0.30      # Wb. Initial tube will have this flux.
MPOL = 3               # No. of Fourier modes to optimise. We set mpol = ntor. 3 is the safest initial guess.

"""" When we create a initial tube surface we need to specify the minor radius.
We must therefore convert from toroidal flux to minor radius. This provides the 
bounds within which to look for the minor_radius that corresponds to the given 
toroidal flux. """
RADIUS_BRACKET = (0.05, 0.60)

#Outside of these minor radii we tend to see degenerate initial qfm surfaces.
TRUSTED_RADII = (0.15, 0.50)

#The resolution of the quadrature grid the QFM residual is evaluated on.
NPHI = NTHETA = 40

#The resolution of the surface on the cross-section plot at phi=0.
PLOT_NTHETA = 400
CUSP_NTHETA = 3000        # samples used to resolve a cusp

# Which cross-sections to show. Units are phi/pi so (0.0, 0.1, 0.2, 0.3) is a four panel plot of the first field period.
PLOT_PHIS_OVER_PI = (0.0, 0.1, 0.2, 0.3)

# The folder name to save surfaces into. They will be saved as "qfm surfaces [config]".
SURFACE_DIR = "qfm surfaces"

# Select the Poincare data to load from poincare_fieldlines.py
POINCARE_N = 80           # no. of field lines in the dataset to load
POINCARE_TMAX = 4000      # tmax of the dataset to load



def new_surface(mpol=MPOL, nphi=NPHI, ntheta=NTHETA):
    """Makes a new surface (unit torus R=1) with the given resolution and grid.

    Phi values are defined as fractions of a full torus so 1/nfp is a single
    field period.
    """
    return SurfaceRZFourier(
        mpol=mpol, ntor=mpol, stellsym=True, nfp=NFP,
        quadpoints_phi=np.linspace(0, 1/NFP, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False))


def initial_surface(axis, minor_radius, mpol=MPOL):
    """Deform the new_surface into a circular tube centered around the magnetic axis.
    flip_theta=True defines the poloidal direction that leads to positive toroidal flux.
    """
    surface = new_surface(mpol)
    surface.fit_to_curve(axis, minor_radius, flip_theta=True)
    return surface


def toroidal_flux(field, surface):
    """Calculate the toroidal flux through the surface cross-section at phi=0."""
    return ToroidalFlux(surface, BiotSavart(field.coils)).J()


def tube_flux(field, axis, minor_radius, mpol=MPOL):
    """Calculate the toroidal flux through the initial surface at phi=0."""
    return toroidal_flux(field, initial_surface(axis, minor_radius, mpol))


def radius_for_flux(field, axis, target_flux, mpol=MPOL,
                    bracket=RADIUS_BRACKET):
    """Find the minor radius that corresponds to a circular tube with the target
    toroidal flux using a scalar root find.

    Returns an error if the minor radius is outside the safe range given in brackets.

    """
    low, high = bracket
    if not tube_flux(field, axis, low, mpol) <= target_flux <= tube_flux(field, axis, high, mpol):
        raise ValueError(
            f"flux {target_flux} Wb is not reachable by a tube of radius "
            f"{low} to {high} m. Grow the surface by flux continuation instead.")

    radius = brentq(
        lambda r: tube_flux(field, axis, r, mpol) - target_flux,
        low, high, xtol=1e-6)

    if not TRUSTED_RADII[0] <= radius <= TRUSTED_RADII[1]:
        print(f"WARNING: minor radius {radius:.4f} is outside the trusted "
              f"range {TRUSTED_RADII[0]}-{TRUSTED_RADII[1]}. "
              f"Grow the surface by flux "
              f"continuation instead.")
    return radius


def cross_section_rz(surface, phi_over_pi=0.0, ntheta=PLOT_NTHETA):
    """Take a cross-sectional slice of a surface at a given toroidal angle.

    The factor of 2 is converting from units of phi/pi to phi/2pi which are
    the units of our quadpoints (fractions of a full toroidal loop).

    Then convert xyz to rz.
    """
    section = SurfaceRZFourier(
        mpol=surface.mpol, ntor=surface.ntor, stellsym=surface.stellsym,
        nfp=surface.nfp, quadpoints_phi=[phi_over_pi/2],
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False))
    section.x = surface.x

    xyz = section.gamma()[0]
    r = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
    z = xyz[:, 2]
    return np.append(r, r[0]), np.append(z, z[0])


def optimise_qfm(field, surface, target_flux, tol=1e-9):
    """Deform the initial surface to minimise the quadratic flux
    through the surface given a toroidal flux constraint.

    Modifies the surface and returns the QFM residual/error.

    Two staged solve: LBFGS is a soft flux constraint,
    SLSQP is an exact flux constraint.
    """
    flux = ToroidalFlux(surface, BiotSavart(field.coils))
    qfm = QfmSurface(BiotSavart(field.coils), surface, flux, target_flux)

    t0 = time.perf_counter()
    qfm.minimize_qfm_penalty_constraints_LBFGS(tol=tol, maxiter=1000,
                                               constraint_weight=1.0)
    t1 = time.perf_counter()
    qfm.minimize_qfm_exact_constraints_SLSQP(tol=tol, maxiter=1000)
    t2 = time.perf_counter()
    print(f"    LBFGS {t1 - t0:.1f} s, SLSQP {t2 - t1:.1f} s")

    return float(np.linalg.norm(qfm.qfm.J()))


def make_qfm_surface(field, axis, target_flux=TARGET_FLUX, mpol=MPOL):
    """Put functions together to make a QFM surface.

    Record the cross-section of the initial tube surface at phi=0.
    """
    minor_radius = radius_for_flux(field, axis, target_flux, mpol) #Find minor radius.
    print(f"minor radius {minor_radius:.4f} m carries {target_flux:.4f} Wb")

    surface = initial_surface(axis, minor_radius, mpol) #Build tube.

    guess = resample_surface(surface) #Record tube cross-section.

    residual = optimise_qfm(field, surface, target_flux) #Deform tube to qfm surface, return residual.

    return surface, guess, toroidal_flux(field, surface), residual


def surface_path(flux, config=CONFIG, nphi=NPHI, ntheta=NTHETA, mpol=MPOL):
    """Determine the folder & filename to give to each surface.
    New folder for each config. Each surface is labelled by its flux, grid and Fourier resolution
    """

    return (Path(f"{SURFACE_DIR} {config}") /
            f"qfm_surface_flux_{flux:.4f}"
            f"_nphi{nphi}_ntheta{ntheta}_mpol{mpol}.json")


def save_surface(surface, flux, config=CONFIG):
    """Save the surface to its corresponding folder."""
    path = surface_path(flux, config, len(surface.quadpoints_phi),
                        len(surface.quadpoints_theta), surface.mpol)
    path.parent.mkdir(parents=True, exist_ok=True)
    surface.save(str(path))
    print(f"saved surface to {path}")
    return path


def plot_all(surfaces_by_flux, poincare_r, poincare_z, poincare_panel, axis,
             initials=(), phis_over_pi=PLOT_PHIS_OVER_PI,
             filename=None, title="", show=True):
    """Plot QFM surfaces over the Poincare data at the specified cross-sections.

    surfaces_by_flux is a list of (flux, surface).
    initials is a list of initial surfaces, drawn dashed.
    phis_over_pi determines the cross-section.
    """
    traced = np.asarray(TRACED_PHIS)
    panel_for = [int(np.argmin(np.abs(traced - p))) for p in phis_over_pi]

    n = len(phis_over_pi)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    figure, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6.5*nrows),
                                squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.axis("off")

    for i, ax in enumerate(flat[:n]):
        phi = phis_over_pi[i]

        here = poincare_panel == panel_for[i]
        ax.scatter(poincare_r[here], poincare_z[here], s=0.25, color="0.45",
                   linewidths=0)

        for flux, surface in surfaces_by_flux:
            r, z = cross_section_rz(surface, phi)
            ax.plot(r, z, "-", linewidth=1.5, label=f"flux = {flux:.3f} Wb")

        for guess in initials:
            r, z = cross_section_rz(guess, phi)
            ax.plot(r, z, "--", linewidth=2, color="k", zorder=4,
                    label="initial guess")

        ax.set_aspect("equal")
        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"phi = {phi:.2f}pi")
        ax.grid(True, linewidth=0.5)

    handles, labels = flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
                  fontsize=9)
    figure.suptitle(title or "QFM surfaces on Poincare sections")
    figure.tight_layout()
    if filename:
        figure.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"saved plot to {filename}")
    if show:
        plt.show()


def resample_surface(surface, nphi=NPHI, ntheta=NTHETA):
    """Make a copy of the surface at a given resolution and grid.
    """
    copy = new_surface(surface.mpol, nphi, ntheta)
    copy.x = surface.x
    return copy


if __name__ == "__main__":

    """Run all the steps to make an initial QFM surface, save it and 
    open a window of the cross-section plot on top of the Poincare scatter data."""
    from poincare_fieldlines import load_poincare_data

    field, axis = build_field(CONFIG)
    surface, guess, flux, residual = make_qfm_surface(field, axis)
    print(f"flux = {flux:.4f} Wb, residual = {residual:.3e}")

    if SAVE:
        save_surface(surface, flux)
        plot_file = f"{CONFIG}_initial_qfm_surface.png"
    else:
        plot_file = None

    r, z, line, panel = load_poincare_data(CONFIG, POINCARE_N, POINCARE_TMAX)
    plot_all([(flux, surface)], r, z, panel, axis,
             initials=[guess],
             phis_over_pi=PLOT_PHIS_OVER_PI,
             filename=plot_file,
             title=f"QFM surface and initial guess, {CONFIG}")

