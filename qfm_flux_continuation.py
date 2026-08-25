"""This script finds QFM surfaces at a given range of fluxes. An "anchor"
surface is loaded and used as the initial surface. From there, we take
the surface and optimise to a new toroidal flux constraint. We follow this
method iteratively to reach every flux in the range.

This method is significantly more successful than naively starting from a circular
cross-section tube at producing QFM surfaces around the island
band.

We load the anchor surface from the output of initial_qfm_surface.py. We then
sweep inwards and outwards in flux from that surface to cover the desired
flux values.

There is an option to save the surfaces and plot the surfaces. 
"""

import numpy as np

from simsopt._core import load

from w7x_config import build_field
from initial_qfm_surface import (CONFIG, NPHI, NTHETA, optimise_qfm,
                                 plot_all, resample_surface, save_surface,
                                 surface_path)
# True means save the surface. False means do not save the surface but still show the plot.
SAVE = True

# Choose the 'anchor' surface from which to make new QFM surfaces at different fluxes.
ANCHOR_FLUX = 0.30

# Which fluxes to calculate QFM surfaces for, in Wb.
TARGET_FLUXES = np.round(np.arange(0.1, 3.401, 0.1), 4)

# Which QFM surfaces to plot.
PLOT_FLUXES = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)

# Which cross-sections to show. Units are phi/pi so (0.0, 0.1, 0.2, 0.3) is a four panel plot of the first field period.
PLOT_PHIS_OVER_PI = (0.0, 0.1, 0.2, 0.3)


def continue_to_flux(field, surface, target_flux):
    """Make a copy of an existing QFM surface and use that as the initial surface to optimise from but at a new flux.

    Return the new optimised QFM surface.
    """
    next_surface = resample_surface(surface, NPHI, NTHETA)
    residual = optimise_qfm(field, next_surface, target_flux)
    print(f"  flux = {target_flux:.4f} Wb, residual = {residual:.3e}")
    return next_surface


def sweep(field, anchor, targets, anchor_flux=ANCHOR_FLUX, save=SAVE):
    """Continue to the fluxes given by sweeping inwards and outwards (up and down in flux) from the anchor surface.

    Returns family which looks like {flux: surface}.
    """
    family = {anchor_flux: anchor}


    for direction in (sorted((t for t in targets if t < anchor_flux), reverse=True),
                      sorted(t for t in targets if t > anchor_flux)):
        surface = anchor
        for target in direction:
            surface = continue_to_flux(field, surface, float(target))
            family[round(float(target), 4)] = surface
            if save:
                save_surface(surface, float(target))

    return family


if __name__ == "__main__":
    from poincare_fieldlines import load_poincare_data

    field, axis = build_field(CONFIG)

    anchor = load(str(surface_path(ANCHOR_FLUX)))
    print(f"anchor: {surface_path(ANCHOR_FLUX)}")

    family = sweep(field, anchor, TARGET_FLUXES)
    print(f"\n{len(family)} surfaces, "
          f"{min(family):.4f} to {max(family):.4f} Wb")

    plot_file = f"{CONFIG}_qfm_family.png" if SAVE else None

    r, z, line, panel = load_poincare_data(CONFIG)
    chosen = [(f, family[f]) for f in PLOT_FLUXES if f in family]
    plot_all(chosen, r, z, panel, axis,
             phis_over_pi=PLOT_PHIS_OVER_PI,
             filename=plot_file,
             title=f"QFM surfaces by flux continuation, {CONFIG} "
                   f"{len(PLOT_PHIS_OVER_PI)} panel plot")
