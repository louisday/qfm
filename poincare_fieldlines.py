"""Poincare plot for a given W7-X coil configuration.

Field line tracing borrowed from Chris Smiet.

Traces field lines through 4 groups of toroidal cross sections.

E.g. Group 0 is phi/pi = 0.0, 0.4, 0.8, 1.2, 1.6 and all the crossings for this cross section are collated onto panel 1 of the plots.

Run this once for a given configuration and then load the saved .npz data with load_poincare_data() in subsequent scripts.
"""

import matplotlib.pyplot as plt
import numpy as np

from simsopt.field import (InterpolatedField, LevelsetStoppingCriterion,
                           SurfaceClassifier, compute_fieldlines)
from simsopt.geo import SurfaceRZFourier

from w7x_config import NFP, build_field

CONFIG = "standard"

NFIELDLINES = 80          # no. of fieldlines, starting from phi=0
R_SPAN = 0.44             # radial distance from the magnetic axis to the last initial field line position, m
TMAX = 4000               # how far each line is traced for
TRACE_TOLERANCE = 1e-9

DEGREE = 4                # interpolation degree
GRID_N = 25               # interpolation cells across the minor radius

# The four cross sections that each get a panel, given in units of pi.
PHIS_OVER_PI = (0.0, 0.1, 0.2, 0.3)


def build_interpolated_field(field, axis):
    """The interpolated field defined inside the device.

    The classifier defines the interpolation grid and then stops escaping field lines.
    """
    boundary = SurfaceRZFourier.from_nphi_ntheta(
        mpol=5, ntor=5, stellsym=True, nfp=NFP,
        range="full torus", nphi=64, ntheta=24)
    boundary.fit_to_curve(axis, 1.60, flip_theta=False)
    classifier = SurfaceClassifier(boundary, h=0.03, p=2)

    r = np.linalg.norm(boundary.gamma()[:, :, 0:2], axis=2)
    z = boundary.gamma()[:, :, 2]

    def skip(r_grid, phi_grid, z_grid):
        """Skip cells that extend outside the boundary wall."""
        points = np.asarray([r_grid, phi_grid, z_grid]).T.copy()
        return list((classifier.evaluate_rphiz(points) < -0.05).flatten())

    print("building the interpolated field")
    interpolated = InterpolatedField(
        field, DEGREE,
        (r.min(), r.max(), GRID_N),        # r range
        (0, 2*np.pi/NFP, 2*GRID_N),        # phi range
        (0, z.max(), GRID_N//2),           # z range, just the positive half
        nfp=NFP, stellsym=True, skip=skip, extrapolate=False)

    return interpolated, classifier


def trace_fieldlines(interpolated, classifier, axis):
    """Trace the field lines through the device. Returns (r, z, line, panel).

    For every cross section, we count a crossing at all the 5 symmetric toroidal positions.
    This gives us 5 crossings for each data set per full toroidal circulation.

    The planes are grouped as [cross section 0 copies, cross section 1 copies, ...].
    So if you want the index of a given crossing within its group then you divide by nfp.
    """
    r_axis, z_axis = axis.gamma()[0, 0], axis.gamma()[0, 2]
    start_r = r_axis + np.linspace(0, R_SPAN, NFIELDLINES)
    start_z = [z_axis] * NFIELDLINES

    phis = [p*np.pi + k*2*np.pi/NFP
            for p in PHIS_OVER_PI for k in range(NFP)]

    print(f"tracing {NFIELDLINES} field lines through {len(phis)} planes")
    _, hits = compute_fieldlines(
        interpolated, list(start_r), start_z,
        tmax=TMAX, tol=TRACE_TOLERANCE, phis=phis,
        stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)])

    r, z, line, panel = [], [], [], []
    for i, crossings in enumerate(hits):
        if len(crossings) == 0:
            continue
        # columns are [t, plane index, x, y, z]
        r.append(np.sqrt(crossings[:, 2]**2 + crossings[:, 3]**2))
        z.append(crossings[:, 4])
        line.append(np.full(len(crossings), i))
        panel.append(crossings[:, 1].astype(int) // NFP)

    return (np.concatenate(r), np.concatenate(z),
            np.concatenate(line), np.concatenate(panel))


def save_poincare_data(r, z, line, panel, config=CONFIG):
    np.savez(f"poincare_{config}.npz", r=r, z=z, line=line, panel=panel)
    print(f"saved {len(r)} points to poincare_{config}.npz")


def load_poincare_data(config=CONFIG):
    """Loads the Poincare data. Returns (r, z, line, panel)."""
    data = np.load(f"poincare_{config}.npz")
    return data["r"], data["z"], data["line"], data["panel"]


def plot_poincare(r, z, line, panel, config=CONFIG):
    """Make a panel for the crossings on each cross section group. Colour code the crossings. Save the plot and then show."""
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))

    for i, ax in enumerate(axes.ravel()):
        here = panel == i
        ax.scatter(r[here], z[here], s=0.3, c=line[here], cmap="jet",
                   linewidths=0)
        ax.set_title(rf"$\phi = {PHIS_OVER_PI[i]:.2f}\pi$")
        ax.set_aspect("equal")
        ax.set_xlabel("r [m]")
        ax.set_ylabel("z [m]")
        ax.grid(True, linewidth=0.3)

    figure.suptitle(f"W7-X {config} configuration")
    figure.tight_layout()
    figure.savefig(f"poincare_{config}.png", dpi=150)
    print(f"saved plot to poincare_{config}.png")
    plt.show()


if __name__ == "__main__":
    field, axis = build_field(CONFIG)
    interpolated, classifier = build_interpolated_field(field, axis)

    r, z, line, panel = trace_fieldlines(interpolated, classifier, axis)
    for i, p in enumerate(PHIS_OVER_PI):
        print(f"  phi = {p}pi: {np.sum(panel == i)} points")

    save_poincare_data(r, z, line, panel)
    plot_poincare(r, z, line, panel)
