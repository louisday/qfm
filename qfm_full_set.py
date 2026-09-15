"""Make a full set of QFM surfaces at the desired fluxes

Run initial_qfm_surface.py first. Its CONFIG, TARGET_FLUX, MPOL and grid
select the initial file automatically. Each direction is swept out and back,
keeping the lower residual at each target. The farthest target is reached
only once. The initial surface is included and saved unchanged in the family.

Set TARGET_FLUXES densely enough for continuation between neighbouring shapes.
All results go to qfm full set [config], the library used by later scripts.
"""

import time

import numpy as np

from simsopt._core import load

from w7x_config import build_field
from initial_qfm_surface import (CONFIG, TARGET_FLUX, MPOL, NPHI, NTHETA,
                                 SURFACE_DIR, initial_surface_path, optimise_qfm,
                                 plot_all, resample_surface, surface_path)

SAVE = True

# These settings are inherited from initial_qfm_surface.py
ANCHOR_FLUX = TARGET_FLUX
ANCHOR_MPOL = MPOL
GRID = NPHI

# Fluxes in Wb. We also include the initial flux by default.
# We need a dense range of fluxes since the method is iterative ie. surfaces must be close for convergence.
TARGET_FLUXES = np.union1d(np.round(np.arange(0.30, 3.4, 0.05), 4),
                           np.round(np.arange(2.10, 2.301, 0.005), 4))

# Choose which QFM surfaces to plot.
PLOT_FLUXES = (0.3,0.4,0.6,0.7, 0.8, 1.0,1.1,1.20,1.30, 1.40,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7 )

# Which toroidal cross-sections to plot. Units are phi/pi so (0.0, 0.1, 0.2, 0.3) is a four panel plot of the first
# field period.
PLOT_PHIS_OVER_PI = (0.0, 0.1, 0.2, 0.3)

# Select the Poincare data to load from poincare_fieldlines.py
POINCARE_N = 20            # no. of field lines in the dataset to load
POINCARE_TMAX = 10000      # tmax of the dataset to load
POINCARE_START_PHI_OVER_PI = 0.0


def output_path(flux, config=CONFIG, nphi=GRID, ntheta=NTHETA, mpol=ANCHOR_MPOL):
    """Determine the name given to each surface."""
    return surface_path(flux, config, nphi, ntheta, mpol, SURFACE_DIR)


def save_surface(surface, flux, config=CONFIG):
    """Save a surface with its correct name."""
    path = output_path(flux, config, len(surface.quadpoints_phi),
                       len(surface.quadpoints_theta), surface.mpol)
    path.parent.mkdir(parents=True, exist_ok=True)
    surface.save(str(path))
    print(f"saved surface to {path}")
    return path


def continue_to_flux(field, surface, target_flux):
    """Make a copy of an existing QFM surface and use that as the initial surface to optimise at a new flux.

    Return the new optimised QFM surface and its residual.
    """
    start = time.perf_counter()
    next_surface = resample_surface(surface, GRID, NTHETA)
    residual = optimise_qfm(field, next_surface, target_flux)
    print(f"  flux = {target_flux:.4f} Wb, residual = {residual:.3e}, "
          f"{time.perf_counter() - start:.1f} s")
    return next_surface, residual


def one_pass(field, surface, targets):
    """Pass through a list of fluxes, optimising at each flux.

    Returns {flux: (surface, residual)}.
    """
    results = {}
    for target in targets:
        surface, residual = continue_to_flux(field, surface, float(target))
        results[round(float(target), 4)] = (surface, residual)
    return results


def sweep(field, anchor, targets, anchor_flux=ANCHOR_FLUX, save=SAVE):
    """Continue to the given fluxes by sweeping inwards and outwards (up and down in flux) from the initial surface.

    Out from the initial surface, then back, keeping the lowest residual surface at each flux.

    Returns a full set of QFM surfaces which looks like {flux: surface}.
    """
    start = time.perf_counter()
    anchor_flux = round(float(anchor_flux), 4)
    targets = sorted({round(float(t), 4) for t in targets})
    family = {anchor_flux: anchor}
    if save:
        save_surface(anchor, anchor_flux)
    returned = 0

    for direction in (sorted((t for t in targets if t < anchor_flux), reverse=True),
                      sorted(t for t in targets if t > anchor_flux)):
        if not direction:
            continue

        print(f"\noutward pass, {len(direction)} fluxes")
        out = one_pass(field, anchor, direction)

        far = round(float(direction[-1]), 4)
        print(f"\nreturn pass from {far:.4f} Wb")
        back = one_pass(field, out[far][0], [t for t in reversed(direction[:-1])])

        for target in direction:
            key = round(float(target), 4)
            surface, residual = out[key]
            if key in back and back[key][1] < residual:
                surface, residual = back[key]
                returned += 1
            family[key] = surface
            if save:
                save_surface(surface, float(target))

    print(f"\n{len(family)} surfaces in {time.perf_counter() - start:.0f} s")
    print(f"the return pass won at {returned} of {len(family) - 1} fluxes")
    return family


if __name__ == "__main__":
    from poincare_fieldlines import load_poincare_data

    field, axis = build_field(CONFIG)

    anchor_path = initial_surface_path()
    if not anchor_path.exists():
        raise SystemExit(f"no initial surface at {anchor_path}\n"
                         "Run initial_qfm_surface.py first with the desired settings.")
    anchor = load(str(anchor_path))
    print(f"anchor:  {anchor_path}")
    print(f"writing: {SURFACE_DIR} {CONFIG}" if SAVE else "not saving")

    # Identify if a surface already exists and you are going to overwrite it.
    if SAVE:
        all_fluxes = np.union1d(TARGET_FLUXES, [ANCHOR_FLUX])
        clashes = [t for t in all_fluxes
                   if output_path(float(t), mpol=ANCHOR_MPOL).exists()]
        if clashes:
            print(f"WARNING: {len(clashes)} of {len(all_fluxes)} targets "
                  f"already exist and will be overwritten:")
            print("  " + ", ".join(f"{float(t):.4f}" for t in clashes))

    family = sweep(field, anchor, TARGET_FLUXES)
    print(f"\n{len(family)} surfaces, "
          f"{min(family):.4f} to {max(family):.4f} Wb")

    plot_file = f"{CONFIG}_qfm_full_set.png" if SAVE else None

    r, z, line, panel = load_poincare_data(
        CONFIG, POINCARE_N, POINCARE_TMAX, POINCARE_START_PHI_OVER_PI
    )
    chosen = [(f, family[f]) for f in PLOT_FLUXES if f in family]
    plot_all(chosen, r, z, panel, axis,
             phis_over_pi=PLOT_PHIS_OVER_PI,
             filename=plot_file,
             title=f"QFM full set, {CONFIG}")
