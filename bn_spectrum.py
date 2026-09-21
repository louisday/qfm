"""Calculate the Fourier coefficients of b.n on a given resonant surface and show on a colour plot.

Calculates the weighted sums:
    A_mn = 2 <b_r sin(2*pi*(m*theta - n*phi))>
    B_mn = 2 <b_r cos(2*pi*(m*theta - n*phi))>

Key parameters:
- the target surface: the high mpol resonant surface that you have calculated a Boozer spline for.
- the training surface: the true Boozer surface that was used in the spline.
- the mode ranges to show in the spectrum.
"""

import matplotlib.pyplot as plt
import numpy as np

from w7x_config import NFP, build_field
from initial_qfm_surface import CONFIG, surface_path
from boozer_to_qfm_spline import angles_path, surface_on_grid

SAVE = True          # True means save the spectrum and the plot. False means plot only.


"""The directory with the resonant surface to use. There must be a spline already made."""
SURFACE_DIR = "qfm full set"

TARGET_FLUX = 2.23
TARGET_MPOL = 25
TARGET_GRID = 200

"""Which training surface spline to use."""
TRAINING_FLUX = 1.7

"""Range of modes to show on plot. The step is nfp due to symmetry."""
M_MAX = 20
N_MAX = 40

"""Set the resonant mode."""
RESONANT_M = 5
RESONANT_N = -5

"""Weight the coordinate map with the Jacobian."""
USE_JACOBIAN = True

"""Repeat field period nfp times. This was only for debugging."""
FULL_TORUS = False


def normal_field(field, surface_file, phis, thetas):
    """Calculate b_r = B.n/B0 on a given surface at given angles.

    Returns (b_r, bn, b0) where b_r is the normalised field bn/b0, bn is the raw
    normal field in tesla, b0 is the average toroidal field over the surface.
    """
    surface = surface_on_grid(surface_file, phis, thetas)
    xyz = surface.gamma()

    field.set_points(xyz.reshape(-1, 3))
    B = field.B().reshape(xyz.shape)
    bn = np.sum(B * surface.unitnormal(), axis=2)

    geometric = np.arctan2(xyz[..., 1], xyz[..., 0])

    """e_phi is the geometric toroidal unit vector."""
    e_phi = np.stack((-np.sin(geometric), np.cos(geometric),
                      np.zeros_like(geometric)), axis=2)
    b0 = float(np.mean(np.sum(B * e_phi, axis=2)))

    return bn / b0, bn, b0


def coefficients(b_r, theta, phi, m_max=M_MAX, n_max=N_MAX, jacobian=None,
                 full_torus=FULL_TORUS):
    """Fourier decompose the b_r to find coefficients of each (m, n) mode.

    This is done by computing the weighted sum:

    A_mn = 2 <b_r sin(2*pi*(m*theta - n*phi)* weight)>

    where angle brackets denote a mean average. The weight comes from the
    relative magnitude of the Jacobian at a given point.
    """
    m_values = np.arange(0, m_max + 1)
    step = 1 if full_torus else NFP
    n_values = np.arange(-n_max, n_max + step, step)

    if full_torus:
        b_r = np.tile(b_r, (NFP, 1))
        theta = np.tile(theta, (NFP, 1))
        phi = np.concatenate([phi + k/NFP for k in range(NFP)], axis=0)
        jacobian = None if jacobian is None else np.tile(jacobian, (NFP, 1))

    weight = 1.0 if jacobian is None else jacobian / np.mean(jacobian)

    A = np.zeros((len(m_values), len(n_values)))
    B = np.zeros_like(A)

    for i, m in enumerate(m_values):
        for j, n in enumerate(n_values):
            phase = 2 * np.pi * (m*theta - n*phi)
            A[i, j] = 2 * np.mean(b_r * np.sin(phase) * weight)
            B[i, j] = 2 * np.mean(b_r * np.cos(phase) * weight)

    # the constant mode has mean square 1, not 1/2, and no sine part
    i0 = int(np.where(m_values == 0)[0][0])
    j0 = int(np.where(n_values == 0)[0][0])
    A[i0, j0] = 0.0
    B[i0, j0] = float(np.mean(b_r * weight))

    return m_values, n_values, A, B


def at_mode(m_values, n_values, values, m, n):
    """Read a single coefficient from a spectrum."""
    return float(values[int(np.where(m_values == m)[0][0]),
                        int(np.where(n_values == n)[0][0])])


def plot_spectrum(m_values, n_values, A, B, title, filename=None):
    """Plot two colour maps of the (m, n) spectrum for sine and cosine coefficients.

    Both panels share a colour scale. Cosine panel should be zero due to stellarator
    symmetry.
    """
    limit = max(np.abs(A).max(), np.abs(B).max())
    extent = (n_values[0] - NFP/2, n_values[-1] + NFP/2,
              m_values[0] - 0.5, m_values[-1] + 0.5)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    for ax, values, name in ((axes[0], A, r"sine  $A_{m,n}/B_0$"),
                             (axes[1], B, r"cosine  $B_{m,n}/B_0$")):
        name = f"{name}      (max {np.abs(values).max():.2e})"
        image = ax.imshow(values, origin="lower", aspect="auto",
                          interpolation="nearest", extent=extent,
                          cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.plot(RESONANT_N, RESONANT_M, "s", mfc="none", mec="black", mew=1.8,
                ms=11)
        ax.set_xticks(n_values[::2])
        ax.set_xlabel(r"toroidal mode number $n$")
        ax.set_ylabel(r"poloidal mode number $m$")
        ax.set_title(name)
        figure.colorbar(image, ax=ax)

    figure.suptitle(title)
    figure.tight_layout()
    if filename:
        figure.savefig(filename, dpi=150)
        print(f"saved plot to {filename}")
    plt.show()


if __name__ == "__main__":


    angles_file = angles_path(TRAINING_FLUX, TARGET_FLUX)
    surface_file = surface_path(TARGET_FLUX, CONFIG, TARGET_GRID, TARGET_GRID,
                                TARGET_MPOL, SURFACE_DIR)
    for path in (angles_file, surface_file):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    angles = np.load(angles_file)
    phi_qfm, theta_qfm = angles["phi_qfm"], angles["theta_qfm"]
    phi_boozer, theta_boozer = angles["phi_boozer"], angles["theta_boozer"]

    print(f"surface {surface_file.name}")
    print(f"angles  {angles_file.name}")
    print(f"grid    {phi_qfm.shape[0]} x {phi_qfm.shape[1]}\n")

    field, axis = build_field(CONFIG)

    b_r, bn, b0 = normal_field(field, surface_file,
                               phi_qfm[:, 0], theta_qfm[0, :])

    print(f"B0            = {b0:+.6f} T   (b_r = B.n / B0, dimensionless)")
    print(f"rms(B.n)      = {np.sqrt(np.mean(bn**2)):.6e} T")
    print(f"max|B.n|      = {np.abs(bn).max():.6e} T")

    jacobian = angles["jacobian"] if USE_JACOBIAN and "jacobian" in angles else None
    if USE_JACOBIAN and jacobian is None:
        print("  Note: no Jacobian in the angle file - re-run "
              "boozer_to_qfm_spline.py\n")

    m_values, n_values, A, B = coefficients(b_r, theta_boozer, phi_boozer,
                                            jacobian=jacobian)

    _, _, A_qfm, B_qfm = coefficients(b_r, theta_qfm, phi_qfm)

    a_res = at_mode(m_values, n_values, A, RESONANT_M, RESONANT_N)
    b_res = at_mode(m_values, n_values, B, RESONANT_M, RESONANT_N)
    a_qfm = at_mode(m_values, n_values, A_qfm, RESONANT_M, RESONANT_N)

    print(f"stellarator symmetry check: max|B_mn| / max|A_mn| = "
          f"{np.abs(B).max()/np.abs(A).max():.2e}   (should be ~1e-13)")

    print(f"\nresonant mode (m, n) = ({RESONANT_M}, {RESONANT_N}), all /B0:")
    print(f"  A_(5,-5) = {a_res:+.6e}  ")
    print(f"  B_(5,-5) = {b_res:+.6e}  ")
    print(f"  amplitude = {np.hypot(a_res, b_res):.6e}, "
          f"phase = {np.degrees(np.arctan2(b_res, a_res)):+.2f} deg")
    print(f"\n  same mode in QFM angles: {a_qfm:+.6e}"
          f"  (ratio {a_res/a_qfm:+.3f})")
    print(f"  largest coefficient anywhere:      {np.abs(A).max():.6e}")

    if SAVE:
        name = f"bn_spectrum_{CONFIG}_flux_{TARGET_FLUX:.4f}_mpol{TARGET_MPOL}.npz"
        np.savez(name, m_values=m_values, n_values=n_values,
                 A_mn=A, B_mn=B, A_mn_qfm=A_qfm, B_mn_qfm=B_qfm,
                 b0=b0, bn=bn, flux=TARGET_FLUX, training_flux=TRAINING_FLUX)
        print(f"\nsaved spectrum to {name}")

    plot_spectrum(m_values, n_values, A, B,
                  f"$B\\cdot n / B_0$ spectrum at $\\Psi_T$ = {TARGET_FLUX:.2f} Wb, "
                  f"mpol={TARGET_MPOL}, Boozer angles from "
                  f"{TRAINING_FLUX:.2f} Wb",
                  filename=(f"{CONFIG}_bn_spectrum_{TARGET_FLUX:.2f}.png"
                            if SAVE else None))
