"""Poincare plot for a given W7-X coil configuration.

Field line tracing borrowed from Chris Smiet.

Traces field lines through 4 groups of toroidal cross-sections.

E.g. Group 1 is phi/pi = 0.0, 0.4, 0.8, 1.2, 1.6 and all the crossings for this cross section are collated onto panel 1 of the plots.

Run this once for a given configuration and then load the saved .npz data with load_poincare_data() in subsequent scripts.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from simsopt.field import (InterpolatedField, LevelsetStoppingCriterion,
                           SurfaceClassifier, compute_fieldlines)
from simsopt.geo import SurfaceRZFourier

from w7x_config import NFP, build_field

CONFIG = "standard"
SAVE = True               # save the data and the plot? False shows the plot only.

NFIELDLINES = 80          # no. of field lines, starting from phi=0
R_SPAN = 0.40             # radial distance from the magnetic axis to the last initial field line position, m
TMAX = 4000               # how far each line is traced for
TRACE_TOLERANCE = 1e-9

DEGREE = 4                # interpolation degree
GRID_N = 25               # interpolation cells across the minor radius

# Which cross-sections to show. Units are phi/pi so (0.0, 0.1, 0.2, 0.3) is a four panel plot of the first field period.
PHIS_OVER_PI = (0.0, 0.1, 0.2, 0.3)


def build_interpolated_field(field, axis):
    """The interpolated field defined inside the device.

    The classifier defines the interpolation grid and then stops escaping field lines.
    """
    start = time.perf_counter()

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

    interpolated = InterpolatedField(
        field, DEGREE,
        (r.min(), r.max(), GRID_N),        # r range
        (0, 2*np.pi/NFP, 2*GRID_N),        # phi range
        (0, z.max(), GRID_N//2),           # z range, just the positive half
        nfp=NFP, stellsym=True, skip=skip, extrapolate=False)
    print(f"interpolated field built in {time.perf_counter() - start:.1f} s")

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

    start = time.perf_counter()
    _, hits = compute_fieldlines(
        interpolated, list(start_r), start_z,
        tmax=TMAX, tol=TRACE_TOLERANCE, phis=phis,
        stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)])
    print(f"traced in {time.perf_counter() - start:.1f} s")

    r, z, line, panel = [], [], [], []
    for i, crossings in enumerate(hits):
        # gets rid of lines that leave the device before crossing a plane.
        crossings = crossings[crossings[:, 1] >= 0]
        if len(crossings) == 0:
            continue

        r.append(np.sqrt(crossings[:, 2]**2 + crossings[:, 3]**2))
        z.append(crossings[:, 4])
        line.append(np.full(len(crossings), i))
        panel.append(crossings[:, 1].astype(int) // NFP)

    return (np.concatenate(r), np.concatenate(z),
            np.concatenate(line), np.concatenate(panel))


def poincare_name(config=CONFIG, nfieldlines=None, tmax=None):
    """Determine the filename to save the data as."""
    if nfieldlines is None:
        nfieldlines = NFIELDLINES
    if tmax is None:
        tmax = TMAX
    return f"poincare_{config}_N{nfieldlines}_tmax{tmax}"


def save_poincare_data(r, z, line, panel, config=CONFIG):
    name = poincare_name(config) + ".npz"
    np.savez(name, r=r, z=z, line=line, panel=panel)
    print(f"saved {len(r)} points to {name}")


def load_poincare_data(config=CONFIG, nfieldlines=None, tmax=None):
    """Loads the Poincare data. Returns (r, z, line, panel)."""
    data = np.load(poincare_name(config, nfieldlines, tmax) + ".npz")
    return data["r"], data["z"], data["line"], data["panel"]


def plot_poincare(r, z, line, panel, config=CONFIG, save=SAVE):
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
    if save:
        name = poincare_name(config) + ".png"
        figure.savefig(name, dpi=150)
        print(f"saved plot to {name}")
    plt.show()


if __name__ == "__main__":
    field, axis = build_field(CONFIG)
    interpolated, classifier = build_interpolated_field(field, axis)

    r, z, line, panel = trace_fieldlines(interpolated, classifier, axis)
    if SAVE:
        save_poincare_data(r, z, line, panel)
    plot_poincare(r, z, line, panel)
