"""Create an angle map taking QFM angles -> Boozer angles for the resonant QFM surface.

The method is as follows:
1. Just below the island region we can find a TRAINING flux where we can make a pair of
smooth, converged surfaces (one QFM and one Boozer). These surfaces should be very
close geometrically.
2. For each QFM point, we find the closest Boozer coordinate (from a dense Boozer grid)
and we construct an angle map based on the difference.
3. We fit a periodic bicubic spline to this angle map.
4. We apply the spline to the TARGET QFM surface (ie the resonant QFM surface) to
find an approximation for a Boozer coordinate parametrisation.

Key parameters:
- the training surfaces: flux, mpol and grid. These surfaces must already exist.
- the target surface: flux, mpol and grid. This surface is the resonant QFM surface and
must already exist.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree

from simsopt._core import load

from w7x_config import NFP
from initial_qfm_surface import CONFIG, surface_path
from boozer_from_qfm import HEADROOM, boozer_path

SAVE = True             # True means save the angles and the plot. False means plot only.
PLOT = True

PERIOD = 1 / NFP        # one field period, in fractions of a full torus


"""The directory, flux and grid of the training pair. There must be a saved QFM surface at mpol and Boozer surface
that was made from this surface."""
SURFACE_DIR = "qfm full set"

TRAINING_FLUX = 1.7
TRAINING_MPOL = 6
TRAINING_GRID = 48

# The resonant QFM surface to map to Boozer surfaces.
TARGET_FLUX = 2.23
TARGET_MPOL = 25
TARGET_GRID = 200

"""How many QFM points to match to Boozer points. This is the grid points
in both toroidal and poloidal directions so 32 x 32. """
LEARN_N = 32


"""The dense grid for the training Boozer surface to compare with the QFM surface."""
MATCH_N = 384

"""The grid to evaluate the coordinate spline on. This matches the grid of the resonant surface. """
TARGET_N = 256


def surface_on_grid(path, phis, thetas):
    """Reload a given surface and sample at a given resolution.

    Creates a copy of a given surface with identical Fourier coefficients
    before sampling the surface at the desired resolution. Works for both
    QFM (SurfaceRZFourier) and Boozer surfaces (SurfaceXYZTensorFourier).
    """
    original = load(str(path))
    copy = type(original)(
        mpol=original.mpol, ntor=original.ntor, stellsym=original.stellsym,
        nfp=original.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    copy.x = original.x
    return copy


def wrap(difference, period):
    """Finds the angular difference in the range [-period/2, period/2)
    ie. taking into account periodic boundaries.

    e.g. If first angle = 0.99 and second angle = 0.01
    then wrap() returns +0.02 rather than 0.98.

    Without this, points matched over a boundary like phi = 0 would be counted
    as ~ a whole field period apart and the spline would fail.
    """
    return (difference + period/2) % period - period/2


def periodic_spline(phis, thetas, values):
    """Makes a smooth function from the QFM angles to the Boozer angles at a given
    flux. Accounts for angle boundaries.

    To ensure the angle boundaries are accounted for we make a copy of our coordinates
    and place these points either side of our period. Think of it like this:

    copy | field period | copy

    Then we fit a cubic interpolation between samples.
    """
    return RectBivariateSpline(
        np.concatenate((phis - PERIOD, phis, phis + PERIOD)),
        np.concatenate((thetas - 1, thetas, thetas + 1)),
        np.tile(values, (3, 3)), kx=3, ky=3, s=0)


def learn_corrections(qfm_file, boozer_file, learn_n=LEARN_N, match_n=MATCH_N):
    """Determines the angle corrections to take QFM coords to Boozer coords at a
    given flux.

    Samples training QFM surface on a LEARN_N grid and the training Boozer surface
    on a much denser grid.

    For each sampled QFM point, finds the nearest Boozer point in physical (x,y,z) space
    using cKDTree to make the search efficient. Then find the correction ie:

    DeltaTheta = theta_boozer - theta_QFM etc.

    Then fit a periodic spline to the coordinate map using periodic_spline().

    The Boozer grid runs over two field periods starting half a field period early so a QFM
    point near the edge of the field period can select a Boozer point just outside it.

    Returns: phi_spline, theta_spline, distance

    Distance is in meters between each pair."""
    phis = np.linspace(0, PERIOD, learn_n, endpoint=False)
    thetas = np.linspace(0, 1, learn_n, endpoint=False)
    xyz_qfm = surface_on_grid(qfm_file, phis, thetas).gamma()

    match_phis = np.linspace(-PERIOD/2, 1.5*PERIOD, 2*match_n, endpoint=False)
    match_thetas = np.linspace(0, 1, match_n, endpoint=False)
    xyz_boozer = surface_on_grid(boozer_file, match_phis, match_thetas).gamma()

    distance, flat = cKDTree(xyz_boozer.reshape(-1, 3)).query(
        xyz_qfm.reshape(-1, 3))
    i_phi, i_theta = np.unravel_index(flat, xyz_boozer.shape[:2])

    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing="ij")
    delta_phi = wrap(match_phis[i_phi].reshape(phi_grid.shape) - phi_grid,
                     PERIOD)
    delta_theta = wrap(match_thetas[i_theta].reshape(theta_grid.shape)
                       - theta_grid, 1.0)

    return (periodic_spline(phis, thetas, delta_phi),
            periodic_spline(phis, thetas, delta_theta),
            distance)


def boozer_angles(phi_spline, theta_spline, phis, thetas):
    """Apply the corrections to a grid of QFM angles.

    ie theta_boozer = theta_QFM + theta_correction

    Since we are making a coordinate mapping we must ensure that we account
    for area scaling in order to avoid uneven weighting across the surface. This
    would skew the "bn_spectrum.py" integral. The Jacobian of our coordinate map
    tells us the local stretching factor.

    J(theta, phi) = (1 + d(dphi)/dphi)(1 + d(dtheta)/dtheta)
              - (d(dphi)/dtheta)(d(dtheta)/dphi)

    If the Jacobian is everywhere positive then the coord map is orientation
    preserving and locally invertible.

    Returns: (phi_grid, theta_grid, phi_boozer, theta_boozer, jacobian).
    """
    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing="ij")

    d_phi_d_phi = phi_spline.ev(phi_grid, theta_grid, dx=1)
    d_phi_d_theta = phi_spline.ev(phi_grid, theta_grid, dy=1)
    d_theta_d_phi = theta_spline.ev(phi_grid, theta_grid, dx=1)
    d_theta_d_theta = theta_spline.ev(phi_grid, theta_grid, dy=1)
    jacobian = ((1.0 + d_phi_d_phi) * (1.0 + d_theta_d_theta)
                - d_phi_d_theta * d_theta_d_phi)

    return (phi_grid, theta_grid,
            phi_grid + phi_spline.ev(phi_grid, theta_grid),
            theta_grid + theta_spline.ev(phi_grid, theta_grid),
            jacobian)


def angles_path(training_flux=None, target_flux=None, config=CONFIG):
    """Determine where to save transferred angles into, and read from.

    Named with both fluxes since they are both important for the spline.
    """
    if training_flux is None:
        training_flux = TRAINING_FLUX
    if target_flux is None:
        target_flux = TARGET_FLUX
    return Path(f"boozer_angles_{config}_from_{training_flux:.4f}"
                f"_to_{target_flux:.4f}.npz")


def plot_corrections(phi_grid, theta_grid, delta_phi, delta_theta,
                     filename=None):
    """Make a two panel colour plots of the angle corrections.
    One for DeltaTheta and one for DeltaPhi.

    Both should look relatively smooth.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, values, name in ((axes[0], 360*delta_phi, r"$\Delta\phi$"),
                             (axes[1], 360*delta_theta, r"$\Delta\theta$")):
        limit = np.abs(values).max()
        mesh = ax.pcolormesh(360*phi_grid, 360*theta_grid, values,
                             cmap="RdBu_r", vmin=-limit, vmax=limit,
                             shading="auto")
        figure.colorbar(mesh, ax=ax, label="degrees")
        ax.set_xlabel(r"QFM $\phi$ [degrees]")
        ax.set_ylabel(r"QFM $\theta$ [degrees]")
        ax.set_title(f"{name}  (max {limit:.3f} deg)")

    figure.suptitle(f"Boozer angle correction carried from "
                    f"{TRAINING_FLUX:.2f} Wb to {TARGET_FLUX:.2f} Wb, {CONFIG}")
    figure.tight_layout()
    if filename:
        figure.savefig(filename, dpi=150)
        print(f"saved plot to {filename}")
    plt.show()


def main():

    """Put the above functions together.

    - Load training and target surfaces
    - Learn angle corrections on training surface
    - Apply angle corrections to the target surface (resonant surface)
    - Print RMS corrections and minimum Jacobian.
    - Save and plot if specified."""

    qfm_file = surface_path(TRAINING_FLUX, CONFIG, TRAINING_GRID,
                            TRAINING_GRID, TRAINING_MPOL, SURFACE_DIR)
    boozer_file = boozer_path(TRAINING_FLUX, CONFIG, TRAINING_GRID,
                              TRAINING_GRID, TRAINING_MPOL, HEADROOM)
    target_file = surface_path(TARGET_FLUX, CONFIG, TARGET_GRID, TARGET_GRID,
                               TARGET_MPOL, SURFACE_DIR)

    for path in (qfm_file, boozer_file, target_file):
        if not path.exists():
            raise SystemExit(f"missing surface: {path}")

    print(f"training on {TRAINING_FLUX:.2f} Wb, mpol={TRAINING_MPOL}")
    print(f"target      {TARGET_FLUX:.2f} Wb, mpol={TARGET_MPOL}\n")

    phi_spline, theta_spline, distance = learn_corrections(
        qfm_file, boozer_file)

    """How far apart in 3D space are the two training surfaces?"""
    print(f"matching distance: mean {1000*distance.mean():.4f} mm, "
          f"max {1000*distance.max():.4f} mm")

    phi_grid, theta_grid, phi_boozer, theta_boozer, jacobian = boozer_angles(
        phi_spline, theta_spline,
        np.linspace(0, PERIOD, TARGET_N, endpoint=False),
        np.linspace(0, 1, TARGET_N, endpoint=False))

    delta_phi = phi_boozer - phi_grid
    delta_theta = theta_boozer - theta_grid
    print(f"RMS corrections: phi {360*np.sqrt(np.mean(delta_phi**2)):.4f} deg, "
          f"theta {360*np.sqrt(np.mean(delta_theta**2)):.4f} deg")
    print(f"minimum coordinate Jacobian: {jacobian.min():.4f}"
          f"   {'OK' if jacobian.min() > 0 else 'Jacobian is negative, coord map is FOLDED'}")

    if SAVE:
        name = angles_path()
        np.savez(name, phi_qfm=phi_grid, theta_qfm=theta_grid,
                 phi_boozer=phi_boozer, theta_boozer=theta_boozer,
                 jacobian=jacobian,
                 training_flux=TRAINING_FLUX, target_flux=TARGET_FLUX,
                 match_distance=distance)
        print(f"\nsaved angles to {name}")

    if PLOT:
        plot_corrections(phi_grid, theta_grid, delta_phi, delta_theta,
                         filename=(f"{CONFIG}_boozer_spline_{TRAINING_FLUX:.2f}"
                                   f"_to_{TARGET_FLUX:.2f}.png") if SAVE else None)


if __name__ == "__main__":
    main()
