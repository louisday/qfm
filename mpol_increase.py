"""This script increases the Fourier modes of a given number of surfaces.

We load a surface, add Fourier modes (set to zero initially) and QFM optimise.
By doing this recursively, we can reach qfm error convergence at around mpol=25.

We recommend the following sequence of increases: mpol = 3 -> 6 -> 7 -> 8 -> ... -> 25.
This can be achieved by setting MPOL_FIRST = 6 and MPOL_MAX = 25.

This ladder approach is much less likely to produce cusps and degenerate surfaces.

We include a speed_ratio for each surface which is a diagnostic designed to allow us to spot
when surfaces have become cusped. The speed ratio should be around 0.3 for a healthy/smooth surface.
"""

import time
import numpy as np

from simsopt._core import load

from w7x_config import build_field
from initial_qfm_surface import (CONFIG, CUSP_NTHETA, NPHI, cross_section_rz,
                                 optimise_qfm, plot_all, resample_surface,
                                 save_surface, surface_path)

# True means save each surface. False means do not save the surface but still show the plot.
SAVE = False

MPOL_SOURCE = 3           # current resolution of the surfaces to use
SOURCE_GRID = 80          # current grid that the surfaces were made on
MPOL_FIRST = 6            # first jump in mpol from the source
MPOL_MAX = 10             # then we increase to this mpol in steps of 1

# Fluxes to increase in mpol. These surfaces must exist at the MPOL_SOURCE and SOURCE_GRID you specify.
FLUXES = (2.30,)

# Select the Poincare data to load from poincare_fieldlines.py
POINCARE_N = 80           # no. of field lines in the dataset to load
POINCARE_TMAX = 4000      # tmax of the dataset to load

# Choose which cross-sections to show. Units are phi/pi so (0.0, 0.1, 0.2, 0.3) is a four panel plot of the first field period.
PLOT_PHIS_OVER_PI = (0.0, 0.1, 0.2, 0.3)

MPOLS = (MPOL_FIRST,) + tuple(range(MPOL_FIRST + 1, MPOL_MAX + 1))


def grid_for(mpol):
    """Define the quadrature grid to use to optimise at. This should scale with mpol.
    """
    return max(NPHI, 8*mpol)


def increase_to(field, surface, flux, mpol):
    """Make new modes up to mpol and then optimise."""
    grid = grid_for(mpol)
    higher = resample_surface(surface.change_resolution(mpol, mpol), grid, grid)
    residual = optimise_qfm(field, higher, flux)
    return higher, residual


def speed_ratio(surface, phi_over_pi, ntheta=CUSP_NTHETA):
    """This tells you the parametrisation speed of each surface at a given phi.
    A value around 0.2 - 1.0 indicates the surface is smooth (with no cusps),
    and a very small value indicates the surface is cusped.

    ie  min|d(r,z)/dtheta| / median(d(r,z)/dtheta)

    """
    r, z = cross_section_rz(surface, phi_over_pi, ntheta)
    r, z = r[:-1], z[:-1]
    speed = np.hypot(np.gradient(r), np.gradient(z))
    return speed.min() / np.median(speed)


if __name__ == "__main__":
    from poincare_fieldlines import load_poincare_data

    field, axis = build_field(CONFIG)
    increased = []

    for flux in FLUXES:
        source = surface_path(flux, nphi=SOURCE_GRID, ntheta=SOURCE_GRID,
                              mpol=MPOL_SOURCE)
        if not source.exists():
            print(f"skipping {flux} Wb, no source surface at {source}")
            continue

        print(f"\n=== flux {flux:.4f} Wb ===")
        surface = load(str(source))

        for mpol in MPOLS:
            grid = grid_for(mpol)
            path = surface_path(flux, nphi=grid, ntheta=grid, mpol=mpol)

            if path.exists():
                surface = load(str(path))
                print(f"  mpol={mpol:2d}  already saved")
                continue

            start = time.perf_counter()
            surface, residual = increase_to(field, surface, flux, mpol)
            print(f"  mpol={mpol:2d}  grid={grid}x{grid}  "
                  f"residual={residual:.3e}  speed={speed_ratio(surface, 0.0):.4f}  "
                  f"{time.perf_counter() - start:.0f} s")

            if SAVE:
                save_surface(surface, flux)

        increased.append((flux, surface))

    plot_file = f"{CONFIG}_qfm_mpol{MPOL_MAX}.png" if SAVE else None

    r, z, line, panel = load_poincare_data(CONFIG, POINCARE_N, POINCARE_TMAX)
    plot_all(increased, r, z, panel, axis,
             phis_over_pi=PLOT_PHIS_OVER_PI,
             filename=plot_file,
             title=f"QFM surfaces increased to mpol={MPOL_MAX}, {CONFIG} "
                   f"{len(PLOT_PHIS_OVER_PI)} panel plot")
