"""Make a QFM surface with a given toroidal flux.

Plot at phi=0:
 - the Poincare scatter data
 - the magnetic axis
 - the initial tube surface cross-section
 - the QFM surface cross-section


Note that this method can produce degenerate surfaces if the toroidal flux is too small or large.
We recommend targeting a flux in the range ~0.2 - 1.5 Wb for the standard configuration of W7-X.

To compute QFM surfaces outside this range it is safer to use the flux continuation script.


"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import brentq

from simsopt.field import BiotSavart
from simsopt.geo import QfmSurface, SurfaceRZFourier, ToroidalFlux

from w7x_config import NFP, build_field

CONFIG = "standard"
SAVE = False             # True means save the surface. False means do not save the surface but still show the plot.

TARGET_FLUX = 0.50      # Wb. Initial tube will have this flux.
MPOL = 3               # No. of Fourier modes to optimise. We set mpol = ntor. 3 is the safest initial guess.

"""" When we create a initial tube surface we need to specify the minor radius.
We must therefore convert from toroidal flux to minor radius. This provides the 
bounds within which to look for the minor_radius that corresponds to the given 
toroidal flux. """
RADIUS_BRACKET = (0.05, 0.60)

#Outside of these minor radii we tend to see degenerate initial qfm surfaces.
TRUSTED_RADII = (0.15, 0.50)

#The resolution of the quadrature grid the QFM residual is evaluated on.
NPHI = NTHETA = 80

#The resolution of the surface on the cross-section plot at phi=0.
PLOT_NTHETA = 400

# The folder name to save surfaces into. They will be saved as "qfm surfaces [config]".ok
SURFACE_DIR = "qfm surfaces"


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

    qfm.minimize_qfm_penalty_constraints_LBFGS(tol=tol, maxiter=1000,
                                               constraint_weight=1.0)
    qfm.minimize_qfm_exact_constraints_SLSQP(tol=tol, maxiter=1000)

    return float(np.linalg.norm(qfm.qfm.J()))


def make_qfm_surface(field, axis, target_flux=TARGET_FLUX, mpol=MPOL):
    """Put functions together to make a QFM surface.

    Record the cross-section of the initial tube surface at phi=0.
    """
    minor_radius = radius_for_flux(field, axis, target_flux, mpol) #Find minor radius.
    print(f"minor radius {minor_radius:.4f} m carries {target_flux:.4f} Wb")

    surface = initial_surface(axis, minor_radius, mpol) #Build tube.

    initial_rz = cross_section_rz(surface) #Record tube cross-section.

    residual = optimise_qfm(field, surface, target_flux) #Deform tube to qfm surface, return residual.

    return surface, initial_rz, toroidal_flux(field, surface), residual


def surface_path(flux, config=CONFIG, nphi=NPHI, ntheta=NTHETA, mpol=MPOL):
    """Determine the folder & filename given to each surface.
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


def plot_all(surfaces_by_flux, poincare_r, poincare_z, axis, initials=(),
             filename=None, title="", show=True):
    """Plot 4 things simultaneously at phi=0:
    - Grey Poincare scatter data.
    - One (or more) QFM surface cross-section(s).
    - Any initial guess surface in a dashed black line.
    - The magnetic axis as a red cross.

    Arguments:
    - surfaces_by_flux is a list of (flux, surface) pairs.
    - poincare_r and z is the Poincare scatter data.
    - axis is the magnetic axis.
    - initials is the initial guess surface cross-section.
    - filename=None means do not save the figure.
    - title="" allows a plot title.
    - show=True shows the plot in a window.

    """
    axis_xyz = axis.gamma()[0]

    plt.figure(figsize=(7, 9))
    plt.scatter(poincare_r, poincare_z, s=0.25, color="0.45", linewidths=0)

    for flux, surface in surfaces_by_flux:
        r, z = cross_section_rz(surface)
        plt.plot(r, z, "-", linewidth=2, label=f"flux = {flux:.3f} Wb")

    for r, z in initials:
        plt.plot(r, z, "--", linewidth=2.5, color="k", zorder=4,
                 label="initial guess")

    plt.plot(np.sqrt(axis_xyz[0]**2 + axis_xyz[1]**2), axis_xyz[2], "rx",
             markersize=12, markeredgewidth=2, zorder=5, label="magnetic axis")

    plt.gca().set_aspect("equal")
    plt.xlabel("r [m]")
    plt.ylabel("z [m]")
    plt.title(title or "QFM surface on a Poincare plot (phi=0)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.grid(True, linewidth=0.5)
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"saved plot to {filename}")
    if show:
        plt.show()


def resample_surface(surface, nphi=NPHI, ntheta=NTHETA):
    """Make a copy of the surface at a given resolution and grid.
    Not used in the initial surface script.
    """
    copy = new_surface(surface.mpol, nphi, ntheta)
    copy.x = surface.x
    return copy


if __name__ == "__main__":

    """Run all the steps to make an initial QFM surface, save it and 
    open a window of the cross-section plot on top of the Poincare scatter data."""
    from poincare_fieldlines import load_poincare_data

    field, axis = build_field(CONFIG)
    surface, initial_rz, flux, residual = make_qfm_surface(field, axis)
    print(f"flux = {flux:.4f} Wb, residual = {residual:.3e}")

    if SAVE:
        save_surface(surface, flux)
        plot_file = f"{CONFIG}_initial_qfm_surface.png"
    else:
        plot_file = None

    r, z, line, panel = load_poincare_data(CONFIG)
    here = panel == 0                       # panel 0 is phi = 0
    plot_all([(flux, surface)], r[here], z[here], axis,
             initials=[initial_rz],
             filename=plot_file,
             title=f"QFM surface and initial guess, {CONFIG} (phi=0)")
